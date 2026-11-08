"""Offline Chapter 7 tests for the locally recreated ``helloagent`` package."""

from pathlib import Path
import os
import unittest
from unittest.mock import patch

import helloagent
from helloagent.agents import MySimpleAgent, ReActAgent, ReflectionAgent
from helloagent.core import Config
from helloagent.tools import ToolRegistry


class FakeLLM:
    """Deterministic LLM replacement; no network or API key is required."""

    provider = "fake"

    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.prompts: list[list[dict[str, str]]] = []

    def invoke(self, messages: list[dict[str, str]], **_kwargs: object) -> str:
        self.prompts.append(messages)
        return next(self.responses)


class Chapter7LocalFrameworkTests(unittest.TestCase):
    def test_imports_resolve_to_workspace_package(self) -> None:
        package_path = Path(helloagent.__file__).resolve()
        workspace_package = Path(__file__).resolve().parents[2] / "src" / "helloagent"

        self.assertTrue(package_path.is_relative_to(workspace_package))
        self.assertNotIn("site-packages", str(package_path))

    def test_registry_runs_a_registered_function(self) -> None:
        registry = ToolRegistry()
        registry.register_function("echo", "Returns input unchanged", lambda text: f"echo:{text}")

        self.assertEqual("echo:hello", registry.execute_tool("echo", "hello"))
        self.assertEqual(["echo"], registry.list_tools())

    def test_config_reads_max_tokens_from_environment(self) -> None:
        with patch.dict(os.environ, {"MAX_TOKENS": "128"}, clear=True):
            config = Config.from_env()

        self.assertEqual(128, config.max_tokens)

    def test_simple_agent_records_local_llm_response(self) -> None:
        agent = MySimpleAgent(name="simple", llm=FakeLLM(["local answer"]))

        self.assertEqual("local answer", agent.run("Say hello"))
        self.assertEqual(["user", "assistant"], [message.role for message in agent.get_history()])

    def test_react_agent_calls_registry_then_finishes(self) -> None:
        registry = ToolRegistry()
        registry.register_function("echo", "Returns input unchanged", lambda text: f"echo:{text}")
        agent = ReActAgent(
            name="react",
            llm=FakeLLM([
                "Thought: I need the tool.\nAction: echo[hello]",
                "Thought: I have the result.\nAction: Finish[Done]",
            ]),
            tool_registry=registry,
        )

        self.assertEqual("Done", agent.run("Use echo"))
        self.assertIn("Observation: echo:hello", agent.current_history)
        self.assertEqual(2, len(agent.get_history()))

    def test_react_agent_accepts_markdown_action_on_following_line(self) -> None:
        registry = ToolRegistry()
        registry.register_function("echo", "Returns input unchanged", lambda text: f"echo:{text}")
        agent = ReActAgent(
            name="react",
            llm=FakeLLM([
                "**Thought:** I need the tool.\n**Action:**\necho[hello]",
                "**Thought:** I have the result.\n**Action:** Finish[Done]",
            ]),
            tool_registry=registry,
        )

        self.assertEqual("Done", agent.run("Use echo"))
        self.assertIn("Observation: echo:hello", agent.current_history)

    def test_reflection_agent_stops_when_feedback_says_no_improvement_needed(self) -> None:
        agent = ReflectionAgent(
            name="reflection",
            llm=FakeLLM(["first answer", "No improvement needed"]),
        )

        self.assertEqual("first answer", agent.run("Write a short answer"))
        self.assertEqual(2, len(agent.get_history()))

    def test_reflection_agent_uses_refinement_prompt(self) -> None:
        agent = ReflectionAgent(
            name="reflection",
            llm=FakeLLM(["first answer", "Add a concrete example", "improved answer"]),
            max_iterations=1,
        )

        self.assertEqual("improved answer", agent.run("Write a short answer"))


if __name__ == "__main__":
    unittest.main()
