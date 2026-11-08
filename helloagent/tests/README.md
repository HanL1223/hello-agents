# Tests

Tests are organised by the production code they cover:

```text
tests/
├── core/       # LLM client and shared framework primitives
└── agents/     # Agent behaviour, including PlanAndSolveAgent
```

Run the complete offline suite from the standalone `helloagent` project root:

```powershell
uv run python -m unittest discover -s tests -v
```

The automated tests use fakes and mocks, so they do not need an API key, a
`.env` file, or network access. A separate manual run of an agent does require
valid LLM environment variables. The manual Plan-and-Solve demo lives at
`tests/manual/plan_solve_demo.py` and can be run with:

```powershell
uv run python tests/manual/plan_solve_demo.py
```

To exercise all local Chapter 7 agent types against the OpenAI API configured
in `.env`, run this paid manual smoke test:

```powershell
uv run python tests/manual/chapter7_openai_smoke.py
```

It runs `MySimpleAgent`, `ReActAgent` with a local `echo` tool,
`ReflectionAgent`, and `PlanAndSolveAgent`. It deliberately excludes the
unfinished Chapter 8 memory system.
