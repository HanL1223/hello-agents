"""Run PlanAndSolveAgent against the LLM configured in the local .env file."""

from dotenv import load_dotenv

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
