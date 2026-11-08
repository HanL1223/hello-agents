"""Working Memory Implementation

- Short-term context management
- Capacity and time constraints
- Priority management
- Automatic cleanup mechanism
"""

from typing import List, Dict, Any
from datetime import datetime, timedelta
import heapq

from ..base import BaseMemory, MemoryItem, MemoryConfig

class WorkingMemory(BaseMemory):
    """Working Memory Implementation

    Features:
    - Limited capacity (typically 10–20 memory items)
    - Short-lived (session-level)
    - Priority management
    - Automatic cleanup of expired memories
    """

    def __init__(self,config:MemoryConfig, storage_backend=None):
        super().__init__(config, self.storage)

        #Workmemory specific config
        self.max_capacity = self.config.working_memory_capacity
        self.max_token = self.config.working_memory_tokens

        # In-memory TTL (minutes); can be overridden by setting
        # working_memory_ttl_minutes in MemoryConfig
        self.max_age_minutes = getattr(self.config,"working_memory_ttl_minutes",120)
        self.current_tokens = 0
        self.session_start = datetime.now()

        # In-memory storage (working memory does not require persistence)
        self.memories: List[MemoryItem] = []

        # Use a priority queue to manage memories
        self.memory_heap = [] # (priority, timestamp, memory_item)

    def add(self,memory_item:MemoryItem) -> str:
        """adding work memory"""

        #Clean expire
        self._expire_old_memories()
        priority = self._calculate_priority(memory_item)

        heapq.heappush(self.memory_heap, (-priority, memory_item.timestamp, memory_item))
        self.memories.append(memory_item)
        
        # update token count
        self.current_tokens += len(memory_item.content.split())
        
        # check capacity
        self._enforce_capacity_limits()
        
        return memory_item.id


    def _expire_old_memories(self):
        """Remove expired memories based on TTL and synchronize the heap and token count."""

        if not self.memories:
            return
        cutoff_time = datetime.now() - timedelta(minutes=self.max_age_minutes)

        #filter preserve memory
        kept: List[MemoryItem] = []
        removed_token_sum = 0
        for m in self.memories:
            if m.timestamp >= cutoff_time:
                kept.append(m)
            else:
                removed_token_sum += len(m.content.split())

        if len(kept) == len(self.memories):
            return

        # Overwrite list and token
        self.memories = kept
        self.current_tokens = max(0,self.current_tokens - self.removed_token_sum)
        # Rebuild heap
        self.memory_heap = []
        for mem in self.memories:
            priority = self._calculate_priority(mem)
            heapq.heappush(self.memory_heap, (-priority, mem.timestamp, mem))

