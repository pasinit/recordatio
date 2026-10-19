# Claude and ChatGPT subscription support (#219)

ChatGPT/Codex support is additive. The existing Claude subscription transport,
CLI-backed identity judge, SDK dependency, and tools remain
supported. Existing deployments continue using Claude without configuration changes.

## Provider selection

- `RECORDATIO_SUBSCRIPTION_PROVIDER=claude` (default): existing Claude agent runtime.
- `RECORDATIO_SUBSCRIPTION_PROVIDER=codex`: Codex CLI authenticated with ChatGPT.
- `RECORDATIO_ENTITY_JUDGE_PROVIDER=claude` (default) or `codex`: independent choice
  for the opt-in identity judge. `RECORDATIO_ENTITY_JUDGE=1` still enables the judge.

The service selects its provider at startup; changing it requires a restart.
Claude retains its existing `agent_rate_limits.db`; Codex uses a separate
`agent_rate_limits_codex.db` so switching providers cannot mix quota state.
Existing Claude credentials and agent configurations are not rewritten.

## Optional Codex service setup

Install the tested CLI version, then sign in as the same OS user that runs
Recordatio. Use a dedicated Codex home: it must contain the service's authentication,
not your personal plugins, MCP servers, hooks, or custom configuration.

```sh
npm install --global @openai/codex@0.154.0
mkdir -p "$HOME/.codex-memento"
CODEX_HOME="$HOME/.codex-memento" codex login
CODEX_HOME="$HOME/.codex-memento" codex login status
```

For a headless host, `codex login --device-auth` is available. Complete the login
in the browser yourself. Do not put OAuth credentials in Recordatio configuration,
agent requests, the repository, or container images.

Set `RECORDATIO_CODEX_HOME` if the service home is elsewhere; `RECORDATIO_CODEX_BIN`
can select an absolute executable path. The Docker image installs the pinned CLI;
mount a persistent service home at `/root/.codex-memento` and authenticate in that
home using the same image/user. A normal `~/.codex` login is separate from this
service login.

## Codex caller configuration

When selecting Codex, configure the caller model to one available in the
ChatGPT Codex account. Leave Claude configurations unchanged for Claude runs. Both a bare model name and
`openai/<model>` are accepted. An empty model uses Codex's default. Claude model
names are rejected only on the Codex path, before starting a turn.

For rollout before reseeding callers, `RECORDATIO_SUBSCRIPTION_MODEL=<Codex model>`
explicitly overrides incoming model names on the subscription path. It also
applies to the identity judge and any explicitly configured subagent models.
Remove the override after updating callers if per-agent model choices are needed.
`RECORDATIO_ENTITY_JUDGE_MODEL` otherwise selects the identity judge model; its
default remains Haiku on Claude and follows Codex on the Codex path.

MCP server names and `mcp__server__tool` allowlists keep their existing format.
HTTP `headers` become Codex `http_headers`; stdio command/args/env pass through.
`allowed_tools: null` exposes configured MCP tools and web search; `[]` disables
all configured MCP servers and web search. Shell execution, browser/computer
control, plugins, apps, and Codex memory are disabled. Filesystem access is
read-only. Codex's built-in orchestration helpers are not a Claude tool sandbox.

`WebSearch` maps to Codex web search. `Task` enables configured Codex subagent
roles. Subagent `name`, `description`, `prompt`, `tools`, and `model` are supported;
role configuration carries each specialist's own MCP allowlist. A null child
tool list inherits the parent; an empty list disables its MCP tools. Claude
filesystem tools such as `Bash`, `Read`, and `Write`, other unsupported built-ins,
extra subagent options, and legacy SSE MCP endpoints produce explicit configuration
errors. Expose needed operations through MCP, or migrate the caller configuration.

Each request uses a fresh ephemeral Codex thread. Recordatio supplies conversation
history and memory briefing on every turn; it does not resume persistent Codex
conversation history. Warm spares are single-use threads sealed to the agent's
config version, with the existing pool capacity and TTL controls.

## Billing and events

The Codex subprocess does not inherit API keys or an OpenAI base-URL override, and
Recordatio verifies that `account/read` reports ChatGPT authentication before a
thread starts. With `RECORDATIO_AGENT_API_FALLBACK` unset, Claude retains its
existing automatic API fallback, while Codex defaults to subscription-only.
Explicit `0` disables API fallback and `1` enables it for the selected provider.
Codex API fallback additionally requires a compatible OpenAI API model/key. The subscription model override does not change
API fallback model selection.

The SSE contract remains `started`, message/thinking deltas and final messages,
MCP tool calls/results, subagent events, rate limits, and one terminal event.
Parent transcripts exclude child-thread messages. Completion token counts split
fresh and cached input into Recordatio's existing buckets. No estimated per-token
USD charge is fabricated for subscription execution. Codex quota notifications
update the rate-state cache; the old per-agent cache is only used for API
fallback when enabled for the selected provider.

Memory extraction also supports `RECORDATIO_EXTRACTION_MODEL=codex/` (CLI default)
or `codex/<model>`. It uses the same dedicated Codex home and does not construct
an API gateway. Explicit API extraction providers remain supported. The general
`/v1/chat/completions` API proxy remains separate from the agent endpoint.

## Cross-repository rollout

Pontifex now provides the shared app-server client, one-shot sessions, and an
OpenAI-shaped utility completion adapter alongside its existing Claude harness.
Recordatio retains its transcript, memory, pool, and SSE integration. Telos selects
Codex for all application model calls and its Compose configuration also selects
Codex for memory extraction. Claude remains selectable.

Deploy the coordinated Pontifex dependency and both applications together.
Authenticate both dedicated service homes before switching the live runtime.
Telos's `CODEX_MIGRATION.md` contains the service login and deployment checklist.
Chat subscriptions do not silently fall back to APIs on authentication or quota
failure. Local tests cover protocol framing, cleanup, tool permissions,
structured results, token usage, and existing Claude behavior; authenticated
live model execution still requires service sign-in.

Tracking: pasinit/recordatio#219, pasinit/pontifex#19, pasinit/telos#228.

Protocol references: [Codex app server](https://developers.openai.com/codex/app-server)
and [Codex configuration](https://developers.openai.com/codex/config-reference).

MCP tools granted by the agent allowlist receive server-scoped Codex approval.
The enabled-tool filter still limits which tools can be called; explicit server
or tool approval overrides remain authoritative. Use
`python scripts/smoke_codex_mcp.py` after deployment to verify real MCP execution
and its SSE events (the probe uses a temporary local MCP server and rate store).
