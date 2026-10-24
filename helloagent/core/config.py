import os
from typing import Optional, Dict, Any
from pydantic import BaseModel

class Config(BaseModel):
    """Config for agent"""

    #LLM Config
    default_model: str = 'gpt-6-luna'
    default_provider: str = 'openai'
    temperature: float= 1.0
    max_token: Optional[int] = None

    #System config
    debug: bool = False
    log_level: str = "INFO"

    #Others
    max_history_length: int = 100

    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables"""
        return cls(
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("MAX_TOKENS")) if os.getenv("MAX_TOKENS") else None,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict"""
        return self.model_dump()