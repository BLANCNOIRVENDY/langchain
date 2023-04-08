"""Chat prompt template."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, List, Sequence, Tuple, Type, Union
import numpy as np

from pydantic import BaseModel, Field

from langchain.memory.buffer import get_buffer_string
from langchain.prompts.base import BasePromptTemplate, StringPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    PromptValue,
    SystemMessage,
)


class BaseMessagePromptTemplate(BaseModel, ABC):
    @abstractmethod
    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        """To messages."""

    @property
    @abstractmethod
    def input_variables(self) -> List[str]:
        """Input variables for this prompt template."""


class MessagesPlaceholder(BaseMessagePromptTemplate):
    """Prompt template that assumes variable is already list of messages."""

    variable_name: str

    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        """To a BaseMessage."""
        value = kwargs[self.variable_name]
        if not isinstance(value, list):
            raise ValueError(
                f"variable {self.variable_name} should be a list of base messages, "
                f"got {value}"
            )
        for v in value:
            if not isinstance(v, BaseMessage):
                raise ValueError(
                    f"variable {self.variable_name} should be a list of base messages,"
                    f" got {value}"
                )
        return value

    @property
    def input_variables(self) -> List[str]:
        """Input variables for this prompt template."""
        return [self.variable_name]


class BaseStringMessagePromptTemplate(BaseMessagePromptTemplate, ABC):
    prompt: StringPromptTemplate
    additional_kwargs: dict = Field(default_factory=dict)

    @classmethod
    def from_template(cls, template: str, **kwargs: Any) -> BaseMessagePromptTemplate:
        prompt = PromptTemplate.from_template(template)
        return cls(prompt=prompt, **kwargs)

    @abstractmethod
    def format(self, **kwargs: Any) -> BaseMessage:
        """To a BaseMessage."""

    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        return [self.format(**kwargs)]

    @property
    def input_variables(self) -> List[str]:
        return self.prompt.input_variables


class ChatMessagePromptTemplate(BaseStringMessagePromptTemplate):
    role: str

    def format(self, **kwargs: Any) -> BaseMessage:
        text = self.prompt.format(**kwargs)
        return ChatMessage(
            content=text, role=self.role, additional_kwargs=self.additional_kwargs
        )


class HumanMessagePromptTemplate(BaseStringMessagePromptTemplate):
    def format(self, **kwargs: Any) -> BaseMessage:
        text = self.prompt.format(**kwargs)
        return HumanMessage(content=text, additional_kwargs=self.additional_kwargs)


class AIMessagePromptTemplate(BaseStringMessagePromptTemplate):
    def format(self, **kwargs: Any) -> BaseMessage:
        text = self.prompt.format(**kwargs)
        return AIMessage(content=text, additional_kwargs=self.additional_kwargs)


class SystemMessagePromptTemplate(BaseStringMessagePromptTemplate):
    def format(self, **kwargs: Any) -> BaseMessage:
        text = self.prompt.format(**kwargs)
        return SystemMessage(content=text, additional_kwargs=self.additional_kwargs)


class ChatPromptValue(PromptValue):
    messages: List[BaseMessage]

    def to_string(self) -> str:
        """Return prompt as string."""
        return get_buffer_string(self.messages)

    def to_messages(self) -> List[BaseMessage]:
        """Return prompt as messages."""
        return self.messages


class ChatPromptTemplate(BasePromptTemplate, ABC):
    input_variables: List[str]
    messages: List[Union[BaseMessagePromptTemplate, BaseMessage]]

    @classmethod
    def from_role_strings(
        cls, string_messages: List[Tuple[str, str]]
    ) -> ChatPromptTemplate:
        messages = [
            ChatMessagePromptTemplate(
                content=PromptTemplate.from_template(template), role=role
            )
            for role, template in string_messages
        ]
        return cls.from_messages(messages)

    @classmethod
    def from_strings(
        cls, string_messages: List[Tuple[Type[BaseMessagePromptTemplate], str]]
    ) -> ChatPromptTemplate:
        messages = [
            role(content=PromptTemplate.from_template(template))
            for role, template in string_messages
        ]
        return cls.from_messages(messages)

    @classmethod
    def from_messages(
        cls, messages: Sequence[Union[BaseMessagePromptTemplate, BaseMessage]]
    ) -> ChatPromptTemplate:
        input_vars = set()
        for message in messages:
            if isinstance(message, BaseMessagePromptTemplate):
                input_vars.update(message.input_variables)
        return cls(input_variables=list(input_vars), messages=messages)

    def format(self, **kwargs: Any) -> str:
        return self.format_prompt(**kwargs).to_string()

    def format_prompt(self, **kwargs: Any) -> PromptValue:
        kwargs = self._merge_partial_and_user_variables(**kwargs)
        result = []
        for message_template in self.messages:
            if isinstance(message_template, BaseMessage):
                result.extend([message_template])
            elif isinstance(message_template, BaseMessagePromptTemplate):
                rel_params = {
                    k: v
                    for k, v in kwargs.items()
                    if k in message_template.input_variables
                }
                message = message_template.format_messages(**rel_params)
                result.extend(message)
            else:
                raise ValueError(f"Unexpected input: {message_template}")
        return ChatPromptValue(messages=result)

    def partial(self, **kwargs: Union[str, Callable[[], str]]) -> BasePromptTemplate:
        raise NotImplementedError

    @property
    def _prompt_type(self) -> str:
        raise NotImplementedError

    def save(self, file_path: Union[Path, str]) -> None:
        raise NotImplementedError


def len_absolute_messages(
        messages:List[Tuple[Union[BaseMessage, BaseMessagePromptTemplate], Any]],
        **kwargs) -> int:
        size = 0
        ws = [w for _,w in messages]
        for message, weight in messages:
            if weight is not None:
                continue
            if isinstance(message, BaseMessage):
                msg = get_buffer_string([message])
                size += len(msg)
            elif isinstance(message, BaseMessagePromptTemplate):
                rel_kwargs = {
                    k:v
                    for k,v in kwargs.items()
                    if k in message.input_variables
                }
                msg = get_buffer_string(message.format_messages(**rel_kwargs))
                size += len(msg)
            else:
                raise ValueError(f"Unexpected input: {message}")
        return size
   
def get_size_capped_message(message:Union[BaseMessage, BaseMessagePromptTemplate], size:int=None, **kwargs) -> List[BaseMessage]:
    if isinstance(message, BaseMessage):
        if size is not None:
            message.limit(size)
            return [message]
        else:
            return [message]
    elif isinstance(message, BaseMessagePromptTemplate):
        rel_kwargs = {
            k:v
            for k,v in kwargs.items() if k in message.input_variables
        }
        if size is None:
            messages = message.format_messages(**rel_kwargs)
            return messages
        else:
            capped_messages = []
            for m in message.format_messages(**rel_kwargs):
                if size <= 0:
                    break
                sized_message = get_size_capped_message(m, size=size, **rel_kwargs)              
                assert len(sized_message) == 1
                size -= (len(get_buffer_string(sized_message)) + len('\n'))
                capped_messages += sized_message
            
            return capped_messages
            
            
         
class SizedChatPromptTemplate(BasePromptTemplate, ABC):
    input_variables: List[str]
    messages: List[Tuple[Union[BaseMessagePromptTemplate, BaseMessage], Any]]
    size: int
    w_sum: float
    
    @classmethod
    def from_messages(cls, messages: List[Union[BaseMessagePromptTemplate, BaseMessage]], weight: List[float], size_cap:int) -> SizedChatPromptTemplate:
        """_summary_

        Args:
            messages (List[Union[BaseMessagePromptTemplate, BaseMessage]]): _description_
            weight (List[float]): when the total number of character is exceeds size_cap, messages will be truncated to fit within the cap, to set weight as None means it should not be truncated.
            size_cap (int): _description_

        Returns:
            SizedChatPromptTemplate: _description_
        """
        assert len(messages) == len(weight)
        w_messages = []
        input_vars = set()
        w_sum = 0
        for m, w in zip(messages, weight):
            w_messages.append((m,w))
            if isinstance(m, BaseMessagePromptTemplate):
                input_vars.update(m.input_variables)
            if w is not None:
                w_sum += w
        return cls(input_variables=list(input_vars), messages=w_messages, size=size_cap, w_sum=w_sum)
    
    def format(self, **kwargs: Any) -> str:
        return self.format_prompt(**kwargs).to_string()
    
    
    def format_prompt(self, **kwargs: Any) -> PromptValue:
        kwargs = self._merge_partial_and_user_variables(**kwargs)
        abs_size = len_absolute_messages(self.messages, **kwargs)
        messages = []
        dyn_size = self.size - abs_size - 1
        for message, weight in self.messages:
            if weight is None:
                messages += get_size_capped_message(message=message, size=None, **kwargs)
            else:
                target_size = int(dyn_size * (weight / self.w_sum)) - 1
                messages += get_size_capped_message(message=message, size=target_size,**kwargs)
        return ChatPromptValue(messages=messages)
        
        