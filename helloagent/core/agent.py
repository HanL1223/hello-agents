from abc import ABC, abstractmethod
from typing import Optional, Any
from .message import Message
from hello_agents import HelloAgentsLLM
from .config import Config


class Agent(ABC):
    """Agent base"""
    
    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()
        self._history: list[Message] = []
    
    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        """runAgent"""
        pass
    
    def add_message(self, message: Message):
        """add mesasge to history"""
        self._history.append(message)
    
    def clear_history(self):
        """clean all history"""
        self._history.clear()
    
    def get_history(self) -> list[Message]:
        """retrive history"""
        return self._history.copy()
    
    def __str__(self) -> str:
        return f"Agent(name={self.name}, provider={self.llm.provider})"
