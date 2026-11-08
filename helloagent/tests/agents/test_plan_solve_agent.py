import unittest

from helloagent.agents.plan_solve_agent import PlanAndSolveAgent, Planner


class FakeLLM:
    """Deterministic LLM substitute: no credentials or network are required."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.prompts: list[list[dict[str, str]]] = []

    def invoke(self, messages: list[dict[str, str]], **_kwargs: object) -> str:
        self.prompts.append(messages)
        return next(self.responses)


class PlanAndSolveAgentTests(unittest.TestCase):
    def test_run_executes_each_planned_step_and_records_history(self) -> None:
        llm = FakeLLM([
            '```python\n["Find Tuesday sales", "Calculate the total"]\n```',
            "Tuesday sales were 30.",
            "The total was 70 apples.",
        ])
        agent = PlanAndSolveAgent(name="test agent", llm=llm)

        answer = agent.run("Monday: 15; Tuesday: double; Wednesday: five less than Tuesday.")

        self.assertEqual("The total was 70 apples.", answer)
        self.assertEqual(3, len(llm.prompts))
        self.assertEqual(2, len(agent.get_history()))
        self.assertEqual("assistant", agent.get_history()[-1].role)

    def test_invalid_plan_stops_without_execution(self) -> None:
        llm = FakeLLM(["This is not a Python code block."])
        agent = PlanAndSolveAgent(name="test agent", llm=llm)

        answer = agent.run("Solve this.")

        self.assertEqual("No valid action plan, abord", answer)
        self.assertEqual(1, len(llm.prompts))
        self.assertEqual(2, len(agent.get_history()))

    def test_planner_returns_only_a_python_list(self) -> None:
        llm = FakeLLM(['```python\n["first", "second"]\n```'])

        self.assertEqual(["first", "second"], Planner(llm).plan("A task"))

    def test_planner_accepts_a_plain_python_list(self) -> None:
        llm = FakeLLM(['["first", "second"]'])

        self.assertEqual(["first", "second"], Planner(llm).plan("A task"))


if __name__ == "__main__":
    unittest.main()
