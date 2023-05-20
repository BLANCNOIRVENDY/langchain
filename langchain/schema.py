"""Common schema objects."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    Generic,
    List,
    NamedTuple,
    Optional,
    Sequence,
    TypeVar,
    Union,
    Tuple)

from pydantic import BaseModel, Extra, Field, root_validator
BUFFER_STRINGIFY_FMT = '{role}: {content}'


def truncate_after_delim(value:str, delim:str) -> Tuple[str,bool]:
    if delim is None:
        raise ValueError(f"delim should not be {delim}")
    last_index = value.rfind(delim)
    if last_index != -1:
        return (value[:last_index + 1], True)
    return (value, False)

def get_buffer_string(
    messages: List[BaseMessage], human_prefix: str = "Human", ai_prefix: str = "AI"
) -> str:
    """Get buffer string of messages."""
    string_messages = []
    for m in messages:
        if isinstance(m, HumanMessage):
            role = human_prefix
        elif isinstance(m, AIMessage):
            role = ai_prefix
        elif isinstance(m, SystemMessage):
            role = "System"
        elif isinstance(m, ChatMessage):
            role = m.role
        else:
            raise ValueError(f"Got unsupported message type: {m}")
        string_messages.append(f"{role}: {m.content}")
    return "\n".join(string_messages)


class AgentAction(NamedTuple):
    """Agent's action to take."""

    tool: str
    tool_input: Union[str, dict]
    log: str


class AgentFinish(NamedTuple):
    """Agent's return value."""

    return_values: dict
    log: str


class Generation(BaseModel):
    """Output of a single generation."""

    text: str
    """Generated text output."""

    generation_info: Optional[Dict[str, Any]] = None
    """Raw generation info response from the provider"""
    """May include things like reason for finishing (e.g. in OpenAI)"""
    # TODO: add log probs


class BaseMessage(BaseModel):
    """Message object."""

    content: str
    additional_kwargs: dict = Field(default_factory=dict)

    @property
    @abstractmethod
    def type(self) -> str:
        """Type of the message, used for serialization."""
    
    def __str__(self) -> str:
        return f'{self.type}: {self.content}'
    
    def limit(self, size_limit:int, delim:str=None) -> Optional[BaseMessage]:
        """
        Limits the size of the message content by cutting it off at a specified size limit.
        If a delimiter is specified, truncates the message content after the last occurrence of the delimiter.

        Args:
            size_limit (int): The maximum size of message content.
            delim (str, optional): The delimiter to be used when truncating the message content. Default value is None.
        """
        content = self.content
        stringified = get_buffer_string([self])
        prefix_size = len(stringified) - len(content)
        content_size = size_limit - prefix_size
        if content_size >= len(content):
            return self
        print(content_size)
        clone = self.copy()
        clone.content = content[:content_size]
        if delim is not None:
            trimmed, success = truncate_after_delim(clone.content, delim=delim)
            if not success:
                return None
            else:
                clone.content = trimmed
        return clone

                
            
           


class HumanMessage(BaseMessage):
    """Type of message that is spoken by the human."""

    example: bool = False

    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "human"


class AIMessage(BaseMessage):
    """Type of message that is spoken by the AI."""

    example: bool = False

    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "ai"


class SystemMessage(BaseMessage):
    """Type of message that is a system message."""

    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "system"


class ChatMessage(BaseMessage):
    """Type of message with arbitrary speaker."""

    role: str

    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "chat"


def _message_to_dict(message: BaseMessage) -> dict:
    return {"type": message.type, "data": message.dict()}


def messages_to_dict(messages: List[BaseMessage]) -> List[dict]:
    return [_message_to_dict(m) for m in messages]


def _message_from_dict(message: dict) -> BaseMessage:
    _type = message["type"]
    if _type == "human":
        return HumanMessage(**message["data"])
    elif _type == "ai":
        return AIMessage(**message["data"])
    elif _type == "system":
        return SystemMessage(**message["data"])
    elif _type == "chat":
        return ChatMessage(**message["data"])
    else:
        raise ValueError(f"Got unexpected type: {_type}")


def messages_from_dict(messages: List[dict]) -> List[BaseMessage]:
    return [_message_from_dict(m) for m in messages]


class ChatGeneration(Generation):
    """Output of a single generation."""

    text = ""
    message: BaseMessage

    @root_validator
    def set_text(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        values["text"] = values["message"].content
        return values


class ChatResult(BaseModel):
    """Class that contains all relevant information for a Chat Result."""

    generations: List[ChatGeneration]
    """List of the things generated."""
    llm_output: Optional[dict] = None
    """For arbitrary LLM provider specific output."""


class LLMResult(BaseModel):
    """Class that contains all relevant information for an LLM Result."""

    generations: List[List[Generation]]
    """List of the things generated. This is List[List[]] because
    each input could have multiple generations."""
    llm_output: Optional[dict] = None
    """For arbitrary LLM provider specific output."""


class PromptValue(BaseModel, ABC):
    @abstractmethod
    def to_string(self) -> str:
        """Return prompt as string."""

    @abstractmethod
    def to_messages(self) -> List[BaseMessage]:
        """Return prompt as messages."""


class BaseMemory(BaseModel, ABC):
    """Base interface for memory in chains."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    @abstractmethod
    def memory_variables(self) -> List[str]:
        """Input keys this memory class will load dynamically."""

    @abstractmethod
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return key-value pairs given the text input to the chain.

        If None, return all memories
        """

    @abstractmethod
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save the context of this model run to memory."""

    @abstractmethod
    def clear(self) -> None:
        """Clear memory contents."""


class BaseChatMessageHistory(ABC):
    """Base interface for chat message history
    See `ChatMessageHistory` for default implementation.
    """

    """
    Example:
        .. code-block:: python

            class FileChatMessageHistory(BaseChatMessageHistory):
                storage_path:  str
                session_id: str

               @property
               def messages(self):
                   with open(os.path.join(storage_path, session_id), 'r:utf-8') as f:
                       messages = json.loads(f.read())
                    return messages_from_dict(messages)

               def add_user_message(self, message: str):
                   message_ = HumanMessage(content=message)
                   messages = self.messages.append(_message_to_dict(_message))
                   with open(os.path.join(storage_path, session_id), 'w') as f:
                       json.dump(f, messages)

               def add_ai_message(self, message: str):
                   message_ = AIMessage(content=message)
                   messages = self.messages.append(_message_to_dict(_message))
                   with open(os.path.join(storage_path, session_id), 'w') as f:
                       json.dump(f, messages)

               def clear(self):
                   with open(os.path.join(storage_path, session_id), 'w') as f:
                       f.write("[]")
    """

    messages: List[BaseMessage]

    @abstractmethod
    def add_user_message(self, message: str) -> None:
        """Add a user message to the store"""

    @abstractmethod
    def add_ai_message(self, message: str) -> None:
        """Add an AI message to the store"""

    @abstractmethod
    def clear(self) -> None:
        """Remove all messages from the store"""
        
class BaseMessageHistoryBackend(ABC):
    
    @abstractmethod
    async def retrieve_history(self, n_history:int) -> List[BaseMessage]:
        """_summary_

        Args:
            n_history (int): _description_

        Raises:
            NotImplementedError: _description_
            NotImplementedError: _description_
            to: _description_

        Returns:
            Awaitable[List[BaseMessage]]: _description_
        """
    
    @abstractmethod
    async def push_history(self, messages: List[BaseMessage]):
        """_summary_

        Args:
            messages (List[BaseMessage]): _description_

        Raises:
            NotImplementedError: _description_
            NotImplementedError: _description_
            to: _description_

        Returns:
            Coroutine: _description_
        """
    
    @abstractmethod
    def close(self):
        """_summary_

        Raises:
            NotImplementedError: _description_
            NotImplementedError: _description_
            to: _description_

        Returns:
            _type_: _description_
        """


class Document(BaseModel):
    """Interface for interacting with a document."""

    page_content: str
    metadata: dict = Field(default_factory=dict)


class BaseRetriever(ABC):
    @abstractmethod
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            List of relevant documents
        """

    @abstractmethod
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            List of relevant documents
        """


# For backwards compatibility


Memory = BaseMemory

T = TypeVar("T")


class BaseOutputParser(BaseModel, ABC, Generic[T]):
    """Class to parse the output of an LLM call.

    Output parsers help structure language model responses.
    """

    @abstractmethod
    def parse(self, text: str) -> T:
        """Parse the output of an LLM call.

        A method which takes in a string (assumed output of a language model )
        and parses it into some structure.

        Args:
            text: output of language model

        Returns:
            structured output
        """

    def parse_with_prompt(self, completion: str, prompt: PromptValue) -> Any:
        """Optional method to parse the output of an LLM call with a prompt.

        The prompt is largely provided in the event the OutputParser wants
        to retry or fix the output in some way, and needs information from
        the prompt to do so.

        Args:
            completion: output of language model
            prompt: prompt value

        Returns:
            structured output
        """
        return self.parse(completion)

    def get_format_instructions(self) -> str:
        """Instructions on how the LLM output should be formatted."""
        raise NotImplementedError

    @property
    def _type(self) -> str:
        """Return the type key."""
        raise NotImplementedError(
            f"_type property is not implemented in class {self.__class__.__name__}."
            " This is required for serialization."
        )

    def dict(self, **kwargs: Any) -> Dict:
        """Return dictionary representation of output parser."""
        output_parser_dict = super().dict()
        output_parser_dict["_type"] = self._type
        return output_parser_dict


class OutputParserException(ValueError):
    """Exception that output parsers should raise to signify a parsing error.

    This exists to differentiate parsing errors from other code or execution errors
    that also may arise inside the output parser. OutputParserExceptions will be
    available to catch and handle in ways to fix the parsing error, while other
    errors will be raised.
    """

    pass

class BasePublisher(ABC):
    
    @abstractmethod
    def publish(self, docs: List[Document]):
        """_summary_

        Args:
            docs (List[Document]): _description_
        """
    
    @abstractmethod
    async def apublish(self,docs:List[Document]):
        """_summary_

        Args:
            docs (List[Document]): _description_
        """

class BaseDocumentTransformer(ABC):
    """Base interface for transforming documents."""

    @abstractmethod
    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Transform a list of documents."""

    @abstractmethod
    async def atransform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Asynchronously transform a list of documents."""
