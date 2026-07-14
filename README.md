# Harness

A small, async agent **harness**: it drives an LLM in a plan-act loop, letting it
call tools until it produces a final answer. Tools can come from three sources —
built-in Python tools, remote [MCP](https://modelcontextprotocol.io) servers, and
in-process skills — all exposed to the model through a single registry.

The planner talks to a local [Ollama](https://ollama.com) model by default via
[litellm](https://github.com/BerriAI/litellm), so the whole thing runs offline
against a model you host yourself.

---

## Features

- **Plan-act orchestration** — an agent loop that plans a step, executes any
  requested tool calls, feeds results back, and repeats until the model responds
  (bounded by `MAX_TOOL_ITERATIONS`).
- **Unified tool registry** — built-in, MCP, and skill tools are listed and
  invoked through one interface and advertised to the model in OpenAI
  function-calling format.
- **Built-in shell tool** — run shell commands with timeout, working-directory,
  and output-truncation controls.
- **MCP client** — connect to servers over streamable HTTP/SSE, with a fast
  preflight reachability check.
- **Per-session memory** — conversation history keyed by session id
  (in-memory by default; pluggable via `BaseMemory`).
- **Typed everywhere** — Pydantic models for messages, tools, and skills;
  structured error codes.

---

## Requirements

- Python **3.12+** (`>=3.12,<3.14`)
- [Poetry](https://python-poetry.org/)
- An [Ollama](https://ollama.com) instance running a tool-capable model
  (only needed to actually run a turn; not required for linting/tests)

---

## Installation

```bash
# Install runtime + dev dependencies and register git hooks
make install

# ...or with Poetry directly
poetry install --with dev
poetry run pre-commit install
```

Poetry is configured to create the virtualenv in-project (`.venv/`).

---

## Configuration

Settings are loaded from the environment (and an optional `.env` file) via
`harness.core.config.Settings`. Nested keys use a `__` delimiter. Defaults:

| Setting | Env var | Default |
| --- | --- | --- |
| LLM model | `OLLAMA_MODEL` | `ollama_chat/qwen3.5:9b-mlx` |
| LLM base URL | `OLLAMA_BASE_URL` | `http://localhost:11434` |
| MCP servers config | `MCP_SERVERS_CONFIG_PATH` | `configs/mcp_servers.yaml` |
| MCP request timeout | `MCP_REQUEST_TIMEOUT_SECONDS` | `30` |
| Log level | `LOG_LEVEL` | `INFO` |

Example `.env`:

```dotenv
OLLAMA_MODEL=ollama_chat/llama3.1:8b
OLLAMA_BASE_URL=http://localhost:11434
LOG_LEVEL=DEBUG
```

### MCP servers (optional)

If you want MCP tools, create the config file referenced by
`MCP_SERVERS_CONFIG_PATH`. It is optional — if the file is missing, the harness
runs with just the built-in tools. The file uses the standard `mcpServers`
mapping and supports two transports:

- **stdio** — a server launched as a local subprocess (`command` + `args`).
- **http** — a remote server over streamable HTTP/SSE (`url`).

```yaml
# configs/mcp_servers.yaml
mcpServers:
  # stdio: spawned as a child process (this is the Playwright MCP server)
  playwright:
    command: npx
    args:
      - "@playwright/mcp@latest"
    # env:            # optional extra env vars for the child process
    #   KEY: value
    # cwd: /path       # optional working directory

  # http: a remote server
  remote:
    url: http://localhost:8080/mcp
```

The transport is chosen per server: an entry with `command` uses stdio, one
with `url` uses http. Servers that fail to connect at startup are logged and
skipped rather than aborting the run.

> **stdio prerequisites:** the `command` must be on the harness's `PATH`.
> Node-based servers like `@playwright/mcp` require **Node.js / `npx`** to be
> installed. The **first** launch of `npx @playwright/mcp@latest` downloads the
> package (and Playwright browsers), which can take longer than the default
> 30s handshake timeout — raise `MCP_REQUEST_TIMEOUT_SECONDS`, or pre-warm it
> once with `npx @playwright/mcp@latest --help`.

---

## Usage

Run the example entry point (sends one prompt and prints the response):

```bash
poetry run python -m harness.main
```

Using the harness programmatically:

```python
import asyncio

from harness.agent.orchestrator.orchestrator import AgentOrchestrator
from harness.core.config import get_settings
from harness.mcp.registry.registry import MCPRegistry
from harness.schemas.chat import ChatMessage
from harness.skills.registry.registry import SkillRegistry
from harness.tools.builtin import ShellTool
from harness.tools.registry import ToolRegistry


async def main() -> None:
    settings = get_settings()

    mcp_registry = MCPRegistry(settings=settings)
    await mcp_registry.initialize()

    tool_registry = ToolRegistry(
        mcp_registry=mcp_registry,
        skill_registry=SkillRegistry(),
    )
    tool_registry.register_builtin_tool(ShellTool())

    orchestrator = AgentOrchestrator(settings=settings, tool_registry=tool_registry)
    try:
        response = await orchestrator.run_turn(
            session_id=None,
            messages=[ChatMessage(role="user", content="What is the current time?")],
        )
        print(response.message.content)
    finally:
        await mcp_registry.shutdown()


asyncio.run(main())
```

### Built-in tools

| Tool | Description |
| --- | --- |
| `shell` | Execute a shell command. Args: `command` (required), `workdir`, `timeout` (seconds, default 120, capped at 600). Returns `exit_code`, `stdout`, `stderr`; output is truncated past 30k characters. |

> ⚠️ The `shell` tool grants the model unrestricted local command execution.
> Only enable it in a trusted or sandboxed environment.

---

## Architecture

```
User messages
     │
     ▼
AgentOrchestrator ── plan ──▶ Planner (FunctionCallingPlanner ▶ litellm ▶ Ollama)
     │  ▲                          │
     │  └──── tool results ────────┘
     ▼
ToolRegistry ──▶ built-in tools (ShellTool, ...)
             ──▶ MCPRegistry ──▶ MCP servers (HTTP/SSE)
             ──▶ SkillRegistry ──▶ skills
     │
     ▼
Memory (per-session history)
```

- **`AgentOrchestrator`** — the plan-act loop; owns memory and the planner.
- **`FunctionCallingPlanner`** — asks the LLM (via litellm) to either answer or
  request tool calls.
- **`ToolRegistry`** — lists/invokes tools from all sources; converts them to
  OpenAI function schemas.
- **`MCPRegistry`** — connects to and multiplexes MCP servers, dispatching to
  the right client per config: `StdioMCPClient` (subprocess) or `HTTPMCPClient`
  (HTTP/SSE), both sharing the transport-agnostic `SessionMCPClient` base.
- **`SkillRegistry` / `BaseSkill`** — in-process skills (extension point; no
  built-in skills ship yet).
- **`BaseMemory` / `InMemoryMemory`** — conversation history storage.

### Project layout

```
harness/
├── agent/
│   ├── memory/         # BaseMemory + in-memory implementation
│   ├── orchestrator/   # AgentOrchestrator (core loop)
│   └── planners/       # BasePlanner + FunctionCallingPlanner
├── core/               # Settings, logging, exceptions
├── mcp/
│   ├── clients/        # BaseMCPClient, SessionMCPClient, stdio + HTTP clients
│   └── registry/       # MCPRegistry
├── schemas/            # Pydantic models (chat, tool, skill)
├── skills/             # BaseSkill + SkillRegistry
├── tools/              # BaseTool, ToolRegistry, built-in tools
├── prompts/            # System prompt(s)
└── main.py             # Example entry point
```

---

## Development

Common tasks are exposed through the `Makefile` (run `make` for the full list):

| Command | Description |
| --- | --- |
| `make install` | Install deps and register git hooks |
| `make format` | Auto-fix import ordering (isort) |
| `make lint` | Run isort (check), flake8, and pylint |
| `make test` | Run the pytest suite |
| `make check` | Lint **and** test (the CI target) |
| `make pre-commit` | Run all pre-commit hooks over every file |
| `make clean` | Remove caches and build artifacts |

### Tooling

- **isort** — import ordering (Black profile), configured in `pyproject.toml`.
- **flake8** — style/lint, configured in `.flake8`.
- **pylint** — static analysis, configured in `pyproject.toml`.
- **pre-commit** — runs the linters plus file-hygiene hooks on every commit
  (`.pre-commit-config.yaml`). All linters run through Poetry so pre-commit and
  `make lint` share the same pinned versions and config.

Line length is **88** across all three tools.

---

## License

Released under the [MIT License](LICENSE). See the `LICENSE` file for the full
text.
