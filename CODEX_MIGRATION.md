# ChatGPT (Codex) subscription support

The hosted agent runtime (`/v1/agents/{user_id}/{slug}/messages`) runs on a
Claude subscription by default. It can instead run on the Codex CLI signed in
with a ChatGPT subscription. Codex support is additive: existing Claude
deployments need no configuration changes.

## Provider selection

| Variable | Values | Effect |
|----------|--------|--------|
| `RECORDATIO_SUBSCRIPTION_PROVIDER` | `claude` (default), `codex` | Agent runtime. Read at startup, so a change needs a restart. |
| `RECORDATIO_ENTITY_JUDGE` | `1` | Enables the opt-in entity identity judge |
| `RECORDATIO_ENTITY_JUDGE_PROVIDER` | `claude` (default), `codex` | Judge provider, chosen independently of the runtime |
| `RECORDATIO_ENTITY_JUDGE_MODEL` | model name | Judge model (default: Haiku on Claude; Codex default on Codex) |
| `RECORDATIO_SUBSCRIPTION_MODEL` | Codex model | Overrides incoming model names on the subscription path, including the judge and subagents. Use it while callers are being migrated. |
| `RECORDATIO_AGENT_API_FALLBACK` | unset, `0`, `1` | Unset: Claude falls back to the API automatically and Codex stays subscription-only. `0` turns fallback off and `1` turns it on. |
| `RECORDATIO_CODEX_HOME` | path | Dedicated Codex home (the Docker image sets `/root/.codex-memento`) |
| `RECORDATIO_CODEX_BIN` | path | Absolute path to the `codex` executable |

Claude and Codex keep separate rate-limit stores (`agent_rate_limits.db` and
`agent_rate_limits_codex.db`), so quota state never mixes.

Memory extraction can also use the subscription:
`RECORDATIO_EXTRACTION_MODEL=codex/` (CLI default model) or `codex/<model>`.

## Service login

Sign in as the OS user that runs Recordatio. Use a dedicated Codex home that
holds only the service's authentication, with no personal plugins, MCP servers
or hooks:

```sh
npm install --global @openai/codex@0.154.0
mkdir -p "$HOME/.codex-memento"
CODEX_HOME="$HOME/.codex-memento" codex login     # or: codex login --device-auth
CODEX_HOME="$HOME/.codex-memento" codex login status
```

Never put OAuth credentials in Recordatio configuration, requests, the
repository or images. In Docker, mount a persistent home at
`/root/.codex-memento` and log in with the same image. Before starting a thread,
Recordatio checks that the account reports ChatGPT authentication. The Codex
subprocess does not inherit API keys or an OpenAI base URL.

## Caller configuration on Codex

- **Models:** set the caller's model to one the ChatGPT account offers, either
  as a bare name or as `openai/<model>`. An empty model uses Codex's default.
  Claude model names are rejected before the turn starts.
- **MCP:** server names and `mcp__server__tool` allowlists are unchanged. HTTP
  `headers` become Codex `http_headers`. `allowed_tools: null` exposes the
  configured MCP tools plus web search, and `[]` disables both. Allowlisted
  tools get server-scoped approval.
- **Built-ins:** `WebSearch` maps to Codex web search. `Task` enables Codex
  subagent roles, which support `name`, `description`, `prompt`, `tools` and
  `model`. Claude filesystem tools (`Bash`, `Read`, `Write`), other unsupported
  built-ins and legacy SSE MCP endpoints are configuration errors.
- **Sandbox:** shell, browser and computer control, plugins, apps and Codex
  memory are disabled. Filesystem access is read-only.
- **Threads:** each request uses a fresh ephemeral thread. Recordatio supplies
  the history and briefing on every turn. Warm spares are single-use threads
  tied to the agent's config version.

The SSE event contract is the same on both providers. Subscription runs report
token counts but no estimated USD cost.

## Verifying

```sh
python scripts/smoke_codex_subscription.py   # live subscription turn
python scripts/smoke_codex_mcp.py            # real MCP tool call and its SSE events
```

References: [Codex app server](https://developers.openai.com/codex/app-server),
[Codex configuration](https://developers.openai.com/codex/config-reference).
