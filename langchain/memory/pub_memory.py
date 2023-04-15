from typing import (
    Dict, 
    Any, 
    List,
    Tuple
)

from pydantic import Field
from langchain.schema import BaseMemory, BasePublisher, BaseMessage, Document
from langchain.prompts.chat import HumanMessage, AIMessage, get_buffer_string
import asyncio

class ChatPairWithMeta:
    chat_pair: Tuple[BaseMessage, BaseMessage]
    meta: Dict[str, Any]
    
    def __init__(self, chat_pair: Tuple[BaseMessage, BaseMessage], **kwargs) -> None:
        self.chat_pair = chat_pair
        self.meta = kwargs
    
    def to_messages(self) -> List[BaseMessage]:
        return [m for m in self.chat_pair]
    
    def to_document(self) -> Document:
        return Document(page_content=get_buffer_string(self.to_messages()), metadata=self.meta)

class ConversationalWindowPublisherMemory(BaseMemory):
    
    publisher: BasePublisher = Field(exclude=True)
    buffer_size: int = 10
    message_pairs: List[ChatPairWithMeta] = []
    memory_key: str
    input_key: str
    output_key: str
    meta_keys: List[str] = []
    return_messages: bool = False
    
    
    def _get_messages(self) -> List[BaseMessage]:
        return [m for m_pair in self.message_pairs for m in m_pair.to_messages()]
    
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        messages = self._get_messages()
        return {self.memory_key: messages if self.return_messages else get_buffer_string(messages=messages)}
    
    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key]
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        input, output = inputs[self.input_key], outputs[self.output_key]
        meta = {
            k:v 
            for k,v in inputs.items() 
            if k in self.meta_keys
        }
        
        self.message_pairs.append(ChatPairWithMeta((HumanMessage(content=input), AIMessage(content=output)),**meta))
        if self.message_pairs and (len(self.message_pairs) > self.buffer_size):
            old, self.message_pairs = self.message_pairs[:self.buffer_size], self.message_pairs[self.buffer_size:]
            asyncio.run(self.publisher.apublish([o.to_document() for o in old]))
    
    def clear(self) -> None:
        return self.messages.clear()
    