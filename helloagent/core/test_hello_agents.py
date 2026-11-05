import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from helloagent.core.hello_agents import HelloAgentsLLM, LLMSettings


class HelloAgentsLLMTests(unittest.TestCase):
    def test_constructor_arguments_override_environment(self) -> None:
        environment = {
            "LLM_PROVIDER": "compatible",
            "LLM_MODEL_ID": "environment-model",
            "LLM_API_KEY": "environment-key",
            "LLM_BASE_URL": "https://environment.example/v1",
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = LLMSettings.resolve(
                model="argument-model",
                api_key="argument-key",
                base_url="https://argument.example/v1",
                provider="compatible",
                timeout=30,
                temperature=0.2,
                max_tokens=42,
            )

        self.assertEqual("argument-model", settings.model)
        self.assertEqual("argument-key", settings.api_key)
        self.assertEqual("https://argument.example/v1", settings.base_url)
        self.assertEqual(30, settings.timeout)
        self.assertEqual(0.2, settings.temperature)
        self.assertEqual(42, settings.max_tokens)

    def test_compatible_provider_requires_base_url(self) -> None:
        with self.assertRaisesRegex(ValueError, "LLM_BASE_URL"):
            LLMSettings.resolve(
                model="test-model",
                api_key="test-key",
                base_url=None,
                provider="compatible",
                timeout=30,
                temperature=0.7,
                max_tokens=None,
            )

    def test_invoke_returns_complete_content(self) -> None:
        llm = HelloAgentsLLM(model="test-model", api_key="test-key")
        llm.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **_: SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content="done"))]
                    )
                )
            )
        )

        self.assertEqual("done", llm.invoke([{"role": "user", "content": "hi"}]))

    def test_stream_invoke_yields_only_text_chunks(self) -> None:
        chunks = [
            SimpleNamespace(choices=[]),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="hel"))]
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=None))]
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="lo"))]
            ),
        ]
        llm = HelloAgentsLLM(model="test-model", api_key="test-key")
        llm.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_: iter(chunks))
            )
        )

        response = llm.stream_invoke([{"role": "user", "content": "hi"}])
        self.assertEqual(["hel", "lo"], list(response))


if __name__ == "__main__":
    unittest.main()
