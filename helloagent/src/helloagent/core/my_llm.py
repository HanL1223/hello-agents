"""Backward-compatible local name for the project's LLM client."""

from .hello_agents import HelloAgentsLLM


class MyLLM(HelloAgentsLLM):
    """Compatibility alias that uses the local OpenAI-compatible client."""


__all__ = ["MyLLM"]
