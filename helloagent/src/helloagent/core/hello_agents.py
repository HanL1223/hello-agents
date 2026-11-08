"""Small, explicit OpenAI-compatible LLM client used by this repository."""

import os
from dataclasses import dataclass
from typing import Any, Iterator, Literal, Optional, Sequence

from openai import OpenAI


SUPPORTED_PROVIDERS = Literal["openai", "compatible"]
MessageInput = Sequence[dict[str, str]]


class HelloAgentsLLMError(RuntimeError):
    """Raised when the LLM client cannot be configured or called."""


@dataclass(frozen=True)
class LLMSettings:
    """Resolved runtime configuration for :class:`HelloAgentsLLM`.

    Precedence is explicit constructor arguments, then ``LLM_*`` variables,
    then the standard OpenAI variables. ``base_url`` is optional for OpenAI,
    because the SDK supplies its normal endpoint when it is omitted.
    """

    model: str
    api_key: str
    base_url: Optional[str]
    provider: SUPPORTED_PROVIDERS
    timeout: float
    temperature: float
    max_tokens: Optional[int]

    @classmethod
    def resolve(
        cls,
        *,
        model: Optional[str],
        api_key: Optional[str],
        base_url: Optional[str],
        provider: Optional[SUPPORTED_PROVIDERS],
        timeout: Optional[float],
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> "LLMSettings":
        resolved_provider = provider or os.getenv("LLM_PROVIDER", "openai")
        if resolved_provider not in {"openai", "compatible"}:
            raise ValueError(
                "LLM_PROVIDER must be 'openai' or 'compatible'; "
                f"received {resolved_provider!r}."
            )

        resolved_model = model or os.getenv("LLM_MODEL_ID")
        resolved_api_key = (
            api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        )
        resolved_base_url = (
            base_url or os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        )
        resolved_timeout = (
            timeout
            if timeout is not None
            else float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
        )
        resolved_temperature = (
            temperature
            if temperature is not None
            else float(os.getenv("LLM_TEMPERATURE", "0.7"))
        )
        resolved_max_tokens = (
            max_tokens
            if max_tokens is not None
            else _optional_int(os.getenv("LLM_MAX_TOKENS"), "LLM_MAX_TOKENS")
        )

        missing = []
        if not resolved_model:
            missing.append("LLM_MODEL_ID")
        if not resolved_api_key:
            missing.append("LLM_API_KEY or OPENAI_API_KEY")
        if resolved_provider == "compatible" and not resolved_base_url:
            missing.append("LLM_BASE_URL")
        if missing:
            raise ValueError("Missing LLM configuration: " + ", ".join(missing))
        if resolved_timeout <= 0:
            raise ValueError("LLM_TIMEOUT_SECONDS must be greater than zero.")
        if not 0 <= resolved_temperature <= 2:
            raise ValueError("LLM_TEMPERATURE must be between 0 and 2.")
        if resolved_max_tokens is not None and resolved_max_tokens <= 0:
            raise ValueError("LLM_MAX_TOKENS must be greater than zero.")

        return cls(
            model=resolved_model,
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            provider=resolved_provider,
            timeout=resolved_timeout,
            temperature=resolved_temperature,
            max_tokens=resolved_max_tokens,
        )


def _optional_int(value: Optional[str], name: str) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer.") from error


class HelloAgentsLLM:
    """A synchronous client for Chat Completions-compatible LLM endpoints.

    Configure the client with constructor arguments or these environment values:
    ``LLM_PROVIDER``, ``LLM_MODEL_ID``, ``LLM_API_KEY``, ``LLM_BASE_URL``,
    ``LLM_TIMEOUT_SECONDS``, ``LLM_TEMPERATURE``, and ``LLM_MAX_TOKENS``.
    ``OPENAI_API_KEY`` and ``OPENAI_BASE_URL`` are accepted as OpenAI aliases.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[SUPPORTED_PROVIDERS] = None,
        timeout: Optional[float] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        self.settings = LLMSettings.resolve(
            model=model,
            api_key=api_key,
            base_url=base_url,
            provider=provider,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.model = self.settings.model
        self.provider = self.settings.provider

        client_options: dict[str, Any] = {
            "api_key": self.settings.api_key,
            "timeout": self.settings.timeout,
        }
        if self.settings.base_url:
            client_options["base_url"] = self.settings.base_url
        self.client = OpenAI(**client_options)

    def invoke(
        self,
        messages: MessageInput,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **options: Any,
    ) -> str:
        """Return one complete assistant response without streaming."""
        try:
            response = self.client.chat.completions.create(
                **self._request_options(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **options,
                )
            )
        except Exception as error:
            raise HelloAgentsLLMError(f"LLM request failed: {error}") from error

        content = response.choices[0].message.content if response.choices else None
        if content is None:
            raise HelloAgentsLLMError("LLM response contained no assistant content.")
        return content

    def stream_invoke(
        self,
        messages: MessageInput,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **options: Any,
    ) -> Iterator[str]:
        """Yield text chunks from one streaming assistant response."""
        try:
            stream = self.client.chat.completions.create(
                **self._request_options(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                    **options,
                )
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as error:
            raise HelloAgentsLLMError(f"LLM streaming request failed: {error}") from error

    def think(
        self,
        messages: MessageInput,
        temperature: Optional[float] = None,
        **options: Any,
    ) -> str:
        """Backward-compatible alias for :meth:`invoke`.

        Use ``stream_invoke`` where streamed chunks are required.
        """
        return self.invoke(messages, temperature=temperature, **options)

    def close(self) -> None:
        """Close the underlying HTTP client when the application shuts down."""
        self.client.close()

    def _request_options(
        self,
        messages: MessageInput,
        *,
        temperature: Optional[float],
        max_tokens: Optional[int],
        **options: Any,
    ) -> dict[str, Any]:
        request_options: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": (
                self.settings.temperature if temperature is None else temperature
            ),
        }
        resolved_max_tokens = (
            self.settings.max_tokens if max_tokens is None else max_tokens
        )
        if resolved_max_tokens is not None:
            # ``max_tokens`` is deprecated and rejected by GPT-6 Chat
            # Completions. Keep the public constructor name for compatibility
            # while sending the current OpenAI request field.
            request_options["max_completion_tokens"] = resolved_max_tokens
        if self.model == "gpt-6-luna":
            # Luna accepts Chat Completions tool/function behaviour only with
            # no reasoning. This also keeps ``temperature`` valid for the
            # Chapter 7 text-protocol agents. Callers can override this via
            # ``invoke(..., reasoning_effort=...)``.
            request_options["reasoning_effort"] = "none"
        request_options.update(options)
        return request_options


def main() -> None:
    """Run a small manual streaming example using the local `.env` file."""
    from dotenv import load_dotenv

    load_dotenv()
    llm = HelloAgentsLLM()
    messages = [
        {"role": "system", "content": "You are a helpful Python assistant."},
        {"role": "user", "content": "Explain a Python class in one sentence."},
    ]
    for chunk in llm.stream_invoke(messages):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
