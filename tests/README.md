# Tests

Tests are organised by the production code they cover:

```text
tests/
├── core/       # LLM client and shared framework primitives
└── agents/     # Agent behaviour, including PlanAndSolveAgent
```

Run the complete offline suite from the repository root:

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
