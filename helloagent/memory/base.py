"""
Memory System Core Classes and Configuration

MemoryItem: Memory item data structure
MemoryConfig: Memory system configuration
BaseMemory: Base class for the memory system
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from datetime import datetime
from pydantic import BaseModel

class MemoryItem(BaseModel):
    """Memory Item Data Structure"""
    id: str
    content: str
    memory_type:str
    user_id:str
    timestamp:datetime
    importance:float = 0.5
    metatdata: Dict[str,Any] = {}

    class Config:
        arbitrary_types_allowed = True

class MemoryConfig(BaseModel):
    """Memory System Configuration"""

    #Storage Path
    storage_path:str = "./memory_data"

    #Status
    max_capacity:int = 100
    importance_threshold: float = 0.1
    decay_factor: float = 0.95

    #working memory specific
    working_memory_capacity: int = 10
    working_memory_tokens: int = 2000
    working_memory_ttl_minutes: int = 120

    # perceptual memory specific
    perceptual_memory_modalities: List[str] = ["text", "image", "audio", "video"]

class BaseMemory(ABC):
    """Base Memory Class

    Defines the common interface and behavior shared by all memory types.
    """

    def __init__(self,config:MemoryConfig, storage_backend = None):
        self.config = config
        self.storage = storage_backend
        self.memory_type = self.__class__.__name__.lower().replace("memory","")

    @abstractmethod
    def add(self,memory_item:MemoryItem) -> str:
        """
        Adding memory item
        
        Args:
            memory_item: object to memorise

        return:
            memory id
        """
        pass

    @abstractmethod
    def retrieve(self,query:str,limit:int = 5, **kwargs) -> List[MemoryItem]:
        """
        Retrieve relevant memories

        Args:
            query: Search query
            limit: Maximum number of memories to return
            **kwargs: Additional retrieval parameters

        Returns:
            A list of relevant memories
        """

        pass

    @abstractmethod
    def update(self, memory_id:str, content:str = None,
               importance: float = None, metadata: Dict[str,Any] = None) -> bool:
        """
        Update a memory

        Args:
            memory_id: Memory ID
            content: Updated content
            importance: Updated importance score
            metadata: Updated metadata

        Returns:
            Whether the update was successful
        """
        pass

    @abstractmethod
    def remove(self, memory_id: str) -> bool:
        """Remove a memory

        Args:
            memory_id: Memory ID

        Returns:
            Whether the deletion was successful
        """
        pass

    @abstractmethod
    def has_memory(self, memory_id: str) -> bool:
        """Check whether a memory exists

        Args:
            memory_id: Memory ID

        Returns:
            Whether the memory exists
        """
        pass

    @abstractmethod
    def clear(self):
        """Clear all memories"""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics

        Returns:
            A dictionary containing memory statistics
        """
        pass

    def _generate_id(self) -> str:
        """Generate memory ID"""
        import uuid
        return str(uuid.uuid4)

    def _calculate_importance(self,content:str, base_importance:float = 0.5) -> float:
        """
        Calculate memory importance

        Args:
            content: Memory content
            base_importance: Base importance score

        Returns:
            Calculated importance score
        """
        importance = base_importance

        #Base on content  length 
        if len(content) > 100:
            importance += .1

        # Keyword-based importance scoring
        important_keywords = ["important", "critical", "must", "note", "warning", "error"]

        if any(keyword in content for keyword in important_keywords):
            importance +=0.2

        return max(0.0, min(1.0, importance))
    
    def __str__(self) -> str:
            stats = self.get_stats()
            return f"{self.__class__.__name__}(count={stats.get('count', 0)})"
    
    def __repr__(self) -> str:
            return self.__str__()