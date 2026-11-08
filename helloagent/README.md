# helloagent

Standalone local project for recreating the Chapter 7 Hello Agents framework.

## Layout

```text
helloagent/
├── src/helloagent/  # local package source
├── tests/           # offline and paid manual tests
├── .env             # local API configuration; never commit
└── pyproject.toml   # only this project's dependencies
```

The project imports `helloagent`, never the copied `hello_agents` package in
the outer course repository.

## Setup

```powershell
cd C:\development\hello-agents\helloagent
uv sync
```

Copy `.env.example` to `.env`, then configure an API key and model.

## Test

Run the offline suite:

```powershell
uv run python -m unittest discover -s tests -v
```

Run the paid Chapter 7 OpenAI smoke test:

```powershell
uv run python tests/manual/chapter7_openai_smoke.py
```

The manual smoke test covers Simple, ReAct, Reflection, and Plan-and-Solve.
It deliberately excludes unfinished Chapter 8 memory work.
