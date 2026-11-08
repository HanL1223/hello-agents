"""Run PlanAndSolveAgent against the LLM configured in the local .env file."""

from pathlib import Path
import sys

from dotenv import load_dotenv

# ``python tests/manual/<script>.py`` makes tests/manual the import root.
# Add this standalone project's src directory so it never relies on the outer
# course repository or the copied reference package.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from helloagent.agents import PlanAndSolveAgent
from helloagent.core.hello_agents import HelloAgentsLLM


def main() -> None:
    load_dotenv()
    llm = HelloAgentsLLM()
    agent = PlanAndSolveAgent(name="Planning and execution assistant", llm=llm)
    question = (
        "A fruit store sold 15 apples on Monday. Tuesday sales were twice Monday's. "
        "Wednesday sales were five fewer than Tuesday's. How many apples were sold in total?"
    )

    result = agent.run(question)
    print(f"\nFinal result: {result}")
    print(f"Conversation history: {len(agent.get_history())} messages")


if __name__ == "__main__":
    main()
