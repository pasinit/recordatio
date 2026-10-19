# agentmem (Recordatio)

The agent runtime supports Claude and, optionally, Codex CLI with a ChatGPT
subscription. Claude remains the default. See [ChatGPT support](CODEX_MIGRATION.md)
for provider selection, authentication, compatibility, and the live smoke test.

**Long-term memory that learns, not just stores.**

An LLM proxy that gives any agent structured, self-improving knowledge — without touching the agent's code.

## The Problem

Agents are stateless across invocations. An agent that has processed 500 SEC filings doesn't perform any differently on filing 501. The memory layer that exists today (vector stores, Mem0, Graphiti) solves storage and retrieval, but not *learning* — extracting patterns from experience and applying them conditionally.

## How It Works

agentmem is an HTTP proxy that sits between your agent and its LLM API. One config change — point your agent at `localhost:8787` instead of the real endpoint.

```
Agent  →  agentmem proxy (localhost:8787)  →  LLM API (Anthropic / OpenAI / Ollama / …)
                ↓                    ↑
          [observe stream]     [inject briefing]
                ↓
          [extract knowledge]
                ↓
          [store & consolidate]
```

### What happens on every request

1. **Briefing injection** — before forwarding, the proxy searches stored knowledge for anything relevant to the current task and prepends it to the system prompt
2. **Transparent forwarding** — the request (including streaming) is forwarded unchanged; the agent sees zero difference
3. **Stream observation** — the proxy accumulates the full response while forwarding every token in real time
4. **Knowledge extraction** — after the turn completes, a background LLM call extracts typed knowledge units (semantic facts, episodic records, procedural strategies)
5. **Storage** — knowledge units are persisted with temporal identity and provenance

## Quick Start

```bash
git clone https://github.com/pasinit/recordatio
cd Recordatio
python3.13 -m venv .venv
.venv/bin/pip install pip-tools
make compile
.venv/bin/pip install -r requirements-dev.txt -e .
```

Create a `.env` file (see [Configuration](#configuration)) then start the proxy:

```bash
make run        # loads .env and starts on port 8787
# or
agentmem --port 8787
```

Point your agent at the proxy:

```bash
# Anthropic SDK
export ANTHROPIC_BASE_URL=http://localhost:8787

# OpenAI SDK / any OpenAI-compatible client
export OPENAI_BASE_URL=http://localhost:8787/v1
```

Identify the memory namespace:

```
x-agent-id: alice
```

If multiple agents share that namespace, send source-agent provenance as well:

```
x-recordatio-source-agent-id: alice/weather
```

That's it. Your agent now has persistent, self-improving memory.

## API

Single LLM endpoint:

```
POST /v1/chat/completions
```

OpenAI-compatible. Works with the OpenAI SDK, httpx, or any client that speaks the OpenAI chat completions format.

### Recordatio-only request fields

These fields are stripped before the request is forwarded upstream:

| Field | Where | Description |
|-------|-------|-------------|
| `x-agent-id` (header) | both | Namespace for memory storage/retrieval. Required for extraction and briefing. |
| `x-recordatio-source-agent-id` (header) | both | Optional provenance label for the agent/conversation that produced extracted memories. Defaults to `x-agent-id` when omitted. |
| `x-recordatio-no-briefing` (body) | both | Set to `true` to skip briefing injection for this request. |
| `x-recordatio-extract-from` (body) | both | Override which messages are sent to the extractor (dict with `last_user_message`). |
| `x-recordatio-skip-input-extraction` (body) | both | Set to `true` to skip the input-side extraction pass. Useful when the user message is a machine-generated trigger (e.g. a scheduled routine) whose content is not worth memorising. |
| `x-recordatio-skip-output-extraction` (body) | both | Set to `true` to skip the output-side extraction pass. The assistant response will not be mined for knowledge. |

### Streaming metadata event

When memories are recalled for a request, the proxy emits a named SSE event at the start of the stream before any model tokens:

```
event: agentmem.briefing
data: {"type": "agentmem.briefing", "units": [...], "entities": [...], "relations": [...], "topic_names": [...]}
```

A Recordatio-aware client can listen for this event to display which memories influenced the response (e.g. "3 memories recalled"). Standard clients (OpenAI SDK, telos, etc.) ignore it because they only process the default unnamed SSE event type.

### Built-in routes

| Path | Description |
|------|-------------|
| `GET /health` | Health check — returns `{"status": "ok"}` |
| `GET /ui` | Agent memory dashboard |
| `GET /memories` | Knowledge graph visualisation |
| `GET /memories/data?agent_id=<id>` | Raw graph data (nodes, edges, units, entities, relations) |
| `GET /memories/data?agent_id=<id>&source_agent_id=<source>` | Raw graph data filtered to memories extracted from one source agent |
| `/mcp` | MCP server (Streamable HTTP). See below. |

## MCP server

The same `agentmem` process exposes Recordatio as an MCP server at `http://localhost:8787/mcp`, alongside the OpenAI-/Anthropic-compatible proxy. Any MCP-capable agent can call its tools directly without routing LLM traffic through Recordatio.

### Tools

| Tool | Description |
|------|-------------|
| `recollect(query)` | Return memories (units, entities, relations, topic hint) relevant to `query` for the calling agent's namespace. |
| `extract(messages, response_text, background=False)` | Run the extraction LLM on both sides of a conversation turn, **persist** the structured results (units / entity mentions / attributes / relation candidates / topics / schema proposals) under the calling agent's namespace, log the cost, and return a typed summary. With `background=True` the work is scheduled fire-and-forget and the tool returns `{"queued": true, "persisted": false}` immediately. |

Both tools return Pydantic-typed payloads (`RecollectResponse` / `ExtractResponse`), so MCP clients see structured JSON with a stable schema.

### Per-agent setup

`agent_id` is taken from the `X-Agent-Id` request header, **not** from a tool argument. The agent operator configures the header once per MCP client; the calling LLM cannot spoof another agent's namespace. HTTP proxy clients may additionally set `X-Recordatio-Source-Agent-Id` to tag extraction provenance within a shared namespace.

```jsonc
// .claude/mcp.json (Claude Code; analogous fields in Claude Desktop / Cursor / SDK clients)
{
  "mcpServers": {
    "recordatio": {
      "type": "http",
      "url": "http://localhost:8787/mcp",
      "headers": {
        "X-Agent-Id": "my-agent/prod"
      }
    }
  }
}
```

Optional per-call headers (set by the client, not the LLM):

| Header | Purpose |
|--------|---------|
| `X-Agent-Id` | **Required.** Memory namespace. |
| `X-Recordatio-Api-Key` | Override the extraction API key (falls back to `RECORDATIO_EXTRACTION_API_KEY`). |
| `X-User-Id` | Tag cost-log rows with a user identifier (mirrors the proxy's convention). |

Extraction performed via the MCP `extract` tool writes to the same `cost_log` table the proxy uses, so MCP-triggered cost is visible in `/costs` alongside proxy-triggered cost.

## Model routing

The proxy uses a `provider/model` convention in the request's `model` field to determine the backend and endpoint. No server-side backend config is needed.

| Prefix | Backend | Default base URL |
|--------|---------|-----------------|
| `anthropic/` | Anthropic SDK | Anthropic's default |
| `openai/` | OpenAI-compatible | OpenAI's default |
| `ollama/` | OpenAI-compatible | `http://localhost:11434/v1` |
| `lmstudio/` | OpenAI-compatible | `http://localhost:1234/v1` |
| `gemini/` | OpenAI-compatible | Google's OAI-compatible endpoint |
| _(no prefix)_ | OpenAI-compatible | OpenAI's default |

Examples:
```
model: "anthropic/claude-opus-4-6"   → Anthropic API
model: "openai/gpt-4o"               → OpenAI API
model: "ollama/qwen2.5:14b"          → local Ollama
model: "gemini/gemini-2.0-flash"     → Google Gemini
model: "gpt-4o"                      → OpenAI API (no prefix fallback)
```

If no model is specified in the request, the proxy returns 400.

## Configuration

All configuration is via environment variables (or a `.env` file in the working directory).

### Extraction LLM

The extraction model runs after each turn to extract knowledge units. It uses the same `provider/model` routing convention.

| Variable | Required | Description |
|----------|----------|-------------|
| `RECORDATIO_EXTRACTION_MODEL` | yes | Extraction model in `provider/model` format |
| `RECORDATIO_EXTRACTION_API_KEY` | no | API key — only needed when the extraction provider requires a different key than the upstream |

### Storage

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENTMEM_DB` | `~/.agentmem/memories.db` | SQLite database path |

### Ontology backfill

Existing databases can be audited and backfilled into the ontology metadata
tables without deleting legacy rows:

```bash
agentmem-backfill --db ~/.agentmem/memories.db --agent-id my-agent/prod --output backfill-audit.json
agentmem-backfill --db ~/.agentmem/memories.db --agent-id my-agent/prod --apply --enable-ontology-apply
```

Dry-run mode is read-only. Apply mode creates a timestamped database backup,
writes the audit JSON, and upserts high-confidence aliases, identifiers,
attributes, entity mentions, and compatibility `mentions` relations.

### Observability

Tracing is disabled by default. Set `OTEL_EXPORTER_OTLP_ENDPOINT` to enable
OpenTelemetry export and to continue incoming `traceparent` context from
upstream callers such as Telos.

| Variable | Default | Description |
|----------|---------|-------------|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | empty | OTLP/gRPC collector endpoint, e.g. `http://localhost:4317` |
| `OTEL_SERVICE_NAME` | `recordatio` | Service name shown in trace backends |

When enabled, Recordatio instruments FastAPI, outbound `httpx` calls, and wraps
Pontifex chat/extraction calls in explicit spans.

### Example: Anthropic upstream + Claude for extraction

```env
RECORDATIO_EXTRACTION_MODEL=anthropic/claude-haiku-4-5-20251001
# RECORDATIO_EXTRACTION_API_KEY= (uses request bearer token)
```

```python
import anthropic
client = anthropic.Anthropic(base_url="http://localhost:8787")
response = client.messages.create(model="anthropic/claude-opus-4-6", ...)
```

### Example: Ollama (local) upstream

```env
RECORDATIO_EXTRACTION_MODEL=ollama/qwen2.5:14b
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8787/v1", api_key="ollama")
response = client.chat.completions.create(model="ollama/qwen2.5:14b", ...)
```

### Example: Ollama upstream, Anthropic for extraction

```env
RECORDATIO_EXTRACTION_MODEL=anthropic/claude-haiku-4-5-20251001
RECORDATIO_EXTRACTION_API_KEY=sk-ant-...
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8787/v1", api_key="ollama")
response = client.chat.completions.create(model="ollama/qwen2.5:14b", ...)
```

### Example: OpenAI upstream

```env
RECORDATIO_EXTRACTION_MODEL=openai/gpt-4o-mini
# RECORDATIO_EXTRACTION_API_KEY= (uses request bearer token)
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8787/v1")
response = client.chat.completions.create(model="openai/gpt-4o", ...)
```

The client's bearer token (`Authorization: Bearer sk-...`) is forwarded to the upstream automatically.

## Knowledge Types

| Type | What | Example |
|------|------|---------|
| **Semantic** | Durable facts about the world | "AAPL Q3 2024 revenue was $94.9B" |
| **Episodic** | Records of specific events | "Compared AAPL and MSFT 10-Ks, found 3 material changes" |
| **Procedural** | Learned strategies and workflows | "Cosine threshold 0.8 reduces false positives on section 1A" |

## Architecture

```
src/tpasini/recordatio/
├── models.py                 # Core data model: KnowledgeUnit, RetrievalContext, Briefing
├── cli.py                    # CLI entry point (agentmem command)
├── entity.py                 # Entity model for the knowledge graph
├── wikidata.py               # Wikidata entity resolution
├── proxy/
│   ├── server.py             # FastAPI proxy: briefing injection, /v1/messages, /v1/chat/completions
│   └── stream_parser.py      # SSE accumulators (Anthropic + OpenAI formats)
├── extraction/
│   ├── extractor.py          # LLM-based knowledge extraction from conversation turns
│   └── consolidation.py      # Episodic → semantic/procedural compression
├── storage/
│   └── base.py               # SQLite store: units, entities, relations, topics
└── ui/
    ├── index.html            # Agent memory dashboard
    └── memories.html         # Knowledge graph visualisation (D3 force graph)
```

### Key design decisions

- **The memory layer owns relevance decisions**, not the agent. The agent never calls `memory.add()` — the proxy observes everything and decides what's worth keeping.
- **Three knowledge types** with different lifecycles: semantic (durable facts), episodic (session records, compressible), procedural (learned strategies, high value).
- **Temporal identity**: every knowledge unit carries `asserted_at`, `valid_from`, `valid_until`. Superseded facts are invalidated, not deleted.
- **Intent-conditioned retrieval**: the briefing is built from the current request context, not a static dump.
- **Provider-agnostic via [Pontifex](https://github.com/pasinit/pontifex)**: the proxy supports Anthropic, OpenAI, Ollama, Gemini, Grok, and any OpenAI-compatible endpoint through a unified streaming interface.
- **Dual wire format**: accepts both Anthropic (`/v1/messages`) and OpenAI (`/v1/chat/completions`) request formats, so any client works without modification.
- **Zero integration effort**: one env var change and an `x-agent-id` header.

## Development

```bash
make lint        # ruff check + mypy
make fix-lint    # auto-fix lint issues
make test        # pytest
make compile     # recompile requirements after editing *.in files
make run         # start the proxy (loads .env)
```

## License

Apache-2.0

The mounted MCP endpoint is `/mcp/mcp`; its session manager starts and stops
with the proxy's parent lifespan. MCP host validation allows loopback and the
Compose address `recordatio:8787` by default. Set `RECORDATIO_MCP_ALLOWED_HOSTS` to a
JSON array of trusted hostnames (including ports) for other deployments. DNS
rebinding protection remains enabled.
