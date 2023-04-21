"""Chat prompt template."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, List, Sequence, Tuple, Type, Union

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


class BaseChatPromptTemplate(BasePromptTemplate, ABC):
    def format(self, **kwargs: Any) -> str:
        return self.format_prompt(**kwargs).to_string()

    def format_prompt(self, **kwargs: Any) -> PromptValue:
        messages = self.format_messages(**kwargs)
        return ChatPromptValue(messages=messages)

    @abstractmethod
    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        """Format kwargs into a list of messages."""


class ChatPromptTemplate(BaseChatPromptTemplate, ABC):
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

    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
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
        return result

    def partial(self, **kwargs: Union[str, Callable[[], str]]) -> BasePromptTemplate:
        raise NotImplementedError

    @property
    def _prompt_type(self) -> str:
        raise NotImplementedError

    def save(self, file_path: Union[Path, str]) -> None:
        raise NotImplementedError

def calculate_message_length(
        messages:List[Tuple[Union[BaseMessage, BaseMessagePromptTemplate], Any]],
        **kwargs) -> Tuple[int,int]:
    """Calculates the total length of messages.
    Calculates the length of messages with a weight of None and messages with a weight.
    Args:
        messages (List[Tuple[Union[BaseMessage, BaseMessagePromptTemplate], Any]]): A list of (message, weight) tuples. Message can be BaseMessage or BaseMessagePromptTemplate.
        **kwargs: Keyword arguments to pass to message formatting.
    Returns:
        Tuple[int, int]: 
        int: The total length of messages with a weight of None.
        int: The total length of messages with a weight.
    """
    unweighted_size = 0
    weighted_size = 0
    for message, weight in messages:
        if isinstance(message, BaseMessage):
            # Get the message as a string using get_buffer_string.
            msg = get_buffer_string([message])
            # Add the length of the message to the total size.
        elif isinstance(message, BaseMessagePromptTemplate):
            # Get only the keyword arguments that are input variables for the template.
            rel_kwargs = {k:v for k,v in kwargs.items() if k in message.input_variables}
            # Format the template messages and get as a string.
            msg = get_buffer_string(message.format_messages(**rel_kwargs))
            # Add the length to the total size.
        else:
            # Invalid message type. Raise an error.
            raise ValueError(f"Unexpected input: {message}")
        if weight is None:
                unweighted_size += len(msg)
        else:
            weighted_size += len(msg)
    return unweighted_size, weighted_size 


def get_size_capped_message(message: Union[BaseMessage, BaseMessagePromptTemplate], size_cap:int=None, delim:str=None, **kwargs) -> List[BaseMessage]:
    """
    Returns a list of messages within the size limit.

    Args:
        message (Union[BaseMessage, BaseMessagePromptTemplate]): An instance of message to be split into message fragments.
        size_cap (int, optional): The maximum size of message part. Default value is None which returns the whole message.
        delim (str, optional): The delimiter to be used when limiting message size. Default value is None.
        **kwargs: If there are input variables defined by inheriting BaseMessagePromptTemplate, you can pass the values using **kwargs. 

    Returns:
        List[BaseMessage]: The list of message fragments within the size limit. If the size limit is not defined, it returns the whole message.
    """

    if isinstance(message, BaseMessage):
        if size_cap is not None:
            message = message.limit(size_cap, delim=delim) # This method limits the message length using the delimiter if it is set.
            return [message] if message is not None else []
        else:
            return [message]

    elif isinstance(message, BaseMessagePromptTemplate):
        rel_kwargs = { # If additional variables are defined in BaseMessagePromptTemplate, extract and apply their values.
            k:v
            for k,v in kwargs.items() if k in message.input_variables
        }

        if size_cap is None: # If there is no size limit, it returns the whole message.
            messages = message.format_messages(**rel_kwargs) 
            return messages
        else:
            messages = []
            uncapped_messages = message.format_messages(**rel_kwargs) # It returns the original message list prior to fragmenting.
            for i in  range(len(uncapped_messages) - 1, 0, -1):
                if size_cap <= 0: # If the size limit is fully applied, exit the loop.
                    break
                capped_message = get_size_capped_message(uncapped_messages[i], size_cap=size_cap, delim=delim,**rel_kwargs) 
                if len(capped_message) == 0:
                    break
                assert len(capped_message) == 1
                size_cap -= (len(get_buffer_string(capped_message)) + len('\n')) # If the message size is chopped, calculate the remaining size.
                messages += capped_message
            messages.reverse() # Reorder the messages in their original order.
            return messages
                
         
class SizedChatPromptTemplate(BasePromptTemplate, ABC):
    input_variables: List[str]
    messages: List[Tuple[Union[BaseMessagePromptTemplate, BaseMessage], Any]]
    size_cap: int
    w_sum: float
    delim: str = None
    
    @classmethod
    def from_messages(cls, messages: List[Union[BaseMessagePromptTemplate, BaseMessage]], weight: List[float], size_cap:int, delim:str=None) -> SizedChatPromptTemplate:
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
        return cls(input_variables=list(input_vars), messages=w_messages, size_cap=size_cap, w_sum=w_sum, delim=delim)
    
    def format(self, **kwargs: Any) -> str:
        return self.format_prompt(**kwargs).to_string()
    
    
    def format_prompt(self, **kwargs: Any) -> PromptValue:
        kwargs = self._merge_partial_and_user_variables(**kwargs)
        unweighted_size, weighted_size = calculate_message_length(self.messages, **kwargs)
        messages = []
        if self.size_cap > (unweighted_size + weighted_size):
            for m, _ in self.messages:
                messages += get_size_capped_message(message=m, size_cap=None, delim=self.delim, **kwargs)
        else:
            dyn_size = self.size_cap - unweighted_size - 1
            for message, weight in self.messages:
                if weight is None:
                    messages += get_size_capped_message(message=message, size_cap=None, delim=self.delim, **kwargs)
                else:
                    target_size = int(dyn_size * (weight / self.w_sum)) - 1
                    messages += get_size_capped_message(message=message, size_cap=target_size, delim=self.delim, **kwargs)
        
        return ChatPromptValue(messages=messages)
        
        