from typing import List

import asyncio
from concurrent.futures import ThreadPoolExecutor

from langchain.schema import (
    AIMessage,
    BaseChatMessageHistory,
    BaseMessage,
    HumanMessage,
    BaseMessageHistoryBackend
)

class BackendSupportedMessageHistory(BaseChatMessageHistory):
    
    @classmethod
    def from_backend(cls,backend: BaseMessageHistoryBackend, n_pair_history:int=5) -> BaseChatMessageHistory:
        loop = asyncio.new_event_loop()
        messages = loop.run_until_complete(backend.retrieve_history(n_history=n_pair_history * 2))
        loop.close()
        return cls(messages=messages, backend=backend, n_pair_history=n_pair_history)
        
    
    def __init__(self, messages:List[BaseMessage], backend: BaseMessageHistoryBackend, n_pair_history:int=5) -> None:
        super().__init__()
        self.n_pair_history = n_pair_history
        self.backend = backend
        self.messages = messages
        
    def add_ai_message(self, message: str) -> None:
        self.messages.append(AIMessage(content=message))
        n_history = self.n_pair_history * 2
        if len(self.messages) > self.n_pair_history * 2:
            self.messages, old = self.messages[-n_history:], self.messages[:-n_history]
            loop = asyncio.new_event_loop()
            loop.run_until_complete(self.backend.push_history(old))
            loop.close()
    
    def add_user_message(self, message: str) -> None:
        self.messages.append(HumanMessage(content=message))
    
    def clear(self) -> None:
        self.messages.clear()

    def __del__(self):
        self.backend.close()