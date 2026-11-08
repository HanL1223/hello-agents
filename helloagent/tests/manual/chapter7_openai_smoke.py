"""Manually exercise every local Chapter 7 agent with the configured OpenAI API.

Run from the standalone project root:
    uv run python tests/manual/chapter7_openai_smoke.py

This is intentionally not part of unittest discovery: it makes paid API calls.
It uses ``LLM_MODEL_ID`` and either ``LLM_API_KEY`` or ``OPENAI_API_KEY`` from
the local ``.env`` file. It does not test Chapter 8 memory or external search.
"""

import os
from pathlib import Path
import sys
from collections.abc import Callable

from dotenv import load_dotenv

# ``python tests/manual/<script>.py`` makes tests/manual the import root.
# Add this standalone project's src directory so it never relies on the outer
# course repository or the copied reference package.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from helloagent.agents import MySimpleAgent, PlanAndSolveAgent, ReActAgent, ReflectionAgent
from helloagent.core import HelloAgentsLLM
from helloagent.tools import ToolRegistry


def require_live_configuration() -> None:
    """Fail before making a request when the required local settings are absent."""
    missing = []
    if not os.getenv("LLM_MODEL_ID"):
        missing.append("LLM_MODEL_ID")
    if not (os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")):
        missing.append("LLM_API_KEY or OPENAI_API_KEY")
    if missing:
        raise SystemExit("Missing .env configuration: " + ", ".join(missing))


def run_case(
    name: str,
    run: Callable[[], str],
    validate: Callable[[str], str | None] | None = None,
) -> bool:
    """Print one readable result without exposing credentials."""
    print(f"\n{'=' * 16} {name} {'=' * 16}")
    try:
        result = run()
    except Exception as error:
        print(f"FAILED: {type(error).__name__}: {error}")
        return False

    if not result or not result.strip():
        print("FAILED: agent returned an empty response")
        return False

    fallback_responses = {
        "Sorry, Mission can't be finish in require steps",
        "No valid action plan, abord",
    }
    if result in fallback_responses:
        print(f"FAILED: agent returned framework fallback: {result}")
        return False

    if validate:
        validation_error = validate(result)
        if validation_error:
            print(f"FAILED: {validation_error}")
            return False

    print(f"Result: {result}")
    return True


def main() -> None:
    load_dotenv()
    require_live_configuration()
    print(f"Using local helloagent with model: {os.environ['LLM_MODEL_ID']}")

    llm = HelloAgentsLLM()
    results: list[bool] = []
    try:
        simple_agent = MySimpleAgent(
            name="Chapter 7 SimpleAgent",
            llm=llm,
            system_prompt="Answer in one sentence.",
        )
        results.append(
            run_case(
                "SimpleAgent",
                lambda: simple_agent.run("What is the purpose of an agent tool registry?"),
            )
        )

        registry = ToolRegistry()
        registry.register_function(
            "echo",
            "Return the input unchanged.",
            lambda text: f"echo:{text}",
        )
        react_agent = ReActAgent(
            name="Chapter 7 ReActAgent",
            llm=llm,
            tool_registry=registry,
            max_steps=3,
        )
        results.append(
            run_case(
                "ReActAgent with local echo tool",
                lambda: react_agent.run(
                    "Use the echo tool with the input 'live smoke test'. "
                    "Then finish with a short confirmation. Follow the Thought/Action format exactly."
                ),
                validate=lambda _result: (
                    None
                    if "Observation: echo:live smoke test" in react_agent.current_history
                    else "the local echo tool was not called successfully"
                ),
            )
        )

        reflection_agent = ReflectionAgent(
            name="Chapter 7 ReflectionAgent",
            llm=llm,
            max_iterations=1,
        )
        results.append(
            run_case(
                "ReflectionAgent",
                lambda: reflection_agent.run(
                    "Write two short sentences explaining why offline tests should run before live API tests."
                ),
            )
        )

        plan_agent = PlanAndSolveAgent(
            name="Chapter 7 PlanAndSolveAgent",
            llm=llm,
        )
        results.append(
            run_case(
                "PlanAndSolveAgent",
                lambda: plan_agent.run(
                    "A store sold 12 books on Monday and twice that many on Tuesday. "
                    "How many books did it sell in total?"
                ),
            )
        )
    finally:
        llm.close()

    passed = sum(results)
    print(f"\nChapter 7 live smoke result: {passed}/{len(results)} agent types passed.")
    if passed != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
