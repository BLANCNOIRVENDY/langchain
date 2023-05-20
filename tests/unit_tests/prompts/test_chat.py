from pathlib import Path
from typing import List

from langchain.prompts import PromptTemplate
from langchain.memory.chat_memory import ChatMessageHistory
from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    BaseMessagePromptTemplate,
    ChatMessagePromptTemplate,
    SizedChatPromptTemplate,
    ChatPromptTemplate,
    ChatPromptValue,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.schema import HumanMessage, SystemMessage


def create_messages() -> List[BaseMessagePromptTemplate]:
    """Create messages."""
    system_message_prompt = SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            template="Here's some context: {context}",
            input_variables=["context"],
        )
    )
    human_message_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template="Hello {foo}, I'm {bar}. Thanks for the {context}",
            input_variables=["foo", "bar", "context"],
        )
    )
    ai_message_prompt = AIMessagePromptTemplate(
        prompt=PromptTemplate(
            template="I'm an AI. I'm {foo}. I'm {bar}.",
            input_variables=["foo", "bar"],
        )
    )
    chat_message_prompt = ChatMessagePromptTemplate(
        role="test",
        prompt=PromptTemplate(
            template="I'm a generic message. I'm {foo}. I'm {bar}.",
            input_variables=["foo", "bar"],
        ),
    )
    return [
        system_message_prompt,
        human_message_prompt,
        ai_message_prompt,
        chat_message_prompt,
    ]


def create_chat_prompt_template() -> ChatPromptTemplate:
    """Create a chat prompt template."""
    return ChatPromptTemplate(
        input_variables=["foo", "bar", "context"],
        messages=create_messages(),
    )


def test_create_chat_prompt_template_from_template() -> None:
    """Create a chat prompt template."""
    prompt = ChatPromptTemplate.from_template("hi {foo} {bar}")
    assert prompt.messages == [
        HumanMessagePromptTemplate.from_template("hi {foo} {bar}")
    ]


def test_create_chat_prompt_template_from_template_partial() -> None:
    """Create a chat prompt template with partials."""
    prompt = ChatPromptTemplate.from_template(
        "hi {foo} {bar}", partial_variables={"foo": "jim"}
    )
    expected_prompt = PromptTemplate(
        template="hi {foo} {bar}",
        input_variables=["bar"],
        partial_variables={"foo": "jim"},
    )
    assert len(prompt.messages) == 1
    output_prompt = prompt.messages[0]
    assert isinstance(output_prompt, HumanMessagePromptTemplate)
    assert output_prompt.prompt == expected_prompt


def test_message_prompt_template_from_template_file() -> None:
    expected = ChatMessagePromptTemplate(
        prompt=PromptTemplate(
            template="Question: {question}\nAnswer:", input_variables=["question"]
        ),
        role="human",
    )
    actual = ChatMessagePromptTemplate.from_template_file(
        Path(__file__).parent.parent / "data" / "prompt_file.txt",
        ["question"],
        role="human",
    )
    assert expected == actual


def test_chat_prompt_template() -> None:
    """Test chat prompt template."""
    prompt_template = create_chat_prompt_template()
    prompt = prompt_template.format_prompt(foo="foo", bar="bar", context="context")
    assert isinstance(prompt, ChatPromptValue)
    messages = prompt.to_messages()
    assert len(messages) == 4
    assert messages[0].content == "Here's some context: context"
    assert messages[1].content == "Hello foo, I'm bar. Thanks for the context"
    assert messages[2].content == "I'm an AI. I'm foo. I'm bar."
    assert messages[3].content == "I'm a generic message. I'm foo. I'm bar."

    string = prompt.to_string()
    expected = (
        "System: Here's some context: context\n"
        "Human: Hello foo, I'm bar. Thanks for the context\n"
        "AI: I'm an AI. I'm foo. I'm bar.\n"
        "test: I'm a generic message. I'm foo. I'm bar."
    )
    assert string == expected

    string = prompt_template.format(foo="foo", bar="bar", context="context")
    assert string == expected


def test_chat_prompt_template_from_messages() -> None:
    """Test creating a chat prompt template from messages."""
    chat_prompt_template = ChatPromptTemplate.from_messages(create_messages())
    assert sorted(chat_prompt_template.input_variables) == sorted(
        ["context", "foo", "bar"]
    )
    assert len(chat_prompt_template.messages) == 4


def test_chat_prompt_template_with_messages() -> None:
    messages = create_messages() + [HumanMessage(content="foo")]
    chat_prompt_template = ChatPromptTemplate.from_messages(messages)
    assert sorted(chat_prompt_template.input_variables) == sorted(
        ["context", "foo", "bar"]
    )
    assert len(chat_prompt_template.messages) == 5
    prompt_value = chat_prompt_template.format_prompt(
        context="see", foo="this", bar="magic"
    )
    prompt_value_messages = prompt_value.to_messages()
    assert prompt_value_messages[-1] == HumanMessage(content="foo")

def test_capped_size_prompt_template_with_messages() -> None:
    SYSTEM_RULE = """You are helpful and intelligent being that help humans to solve their problems. the qualities of top priority you should follow are 
    - Accuracy
    - Adaptability
    - Helpful
    - Reliability
    - Harmlessness
    - Honesty
    """

    USER_MESSAGE = "Who is sakamoto ryuichi?"
    BOT_MESSAGE = """
    Ryuichi Sakamoto is a Japanese musician, composer, and actor. He was born on January 17, 1952, in Tokyo, Japan. Sakamoto began his career as a member of the electronic music group Yellow Magic Orchestra in the late 1970s. He is known for his innovative music compositions that blend electronic sounds with traditional Japanese music, as well as his contributions to the film industry as a film composer. Some of Sakamoto's notable works include the scores for the movies "The Last Emperor," "Merry Christmas, Mr. Lawrence," and "The Revenant." Sakamoto has won numerous awards for his work over the years, including an Academy Award, two Golden Globe Awards, a Grammy Award, and a BAFTA Award.
    """


    history = ChatMessageHistory()
    for _ in range(10):
        history.add_user_message(USER_MESSAGE)
        history.add_ai_message(BOT_MESSAGE)
        
    capped_chat_prompt = SizedChatPromptTemplate.from_messages([
        SystemMessage(content=SYSTEM_RULE),
        MessagesPlaceholder(variable_name='history'),
        HumanMessagePromptTemplate.from_template("{input}")
    ], [None, 1.0, None], 4096)

    capped_prompt = capped_chat_prompt.format(input="What is Large Language Model?", history=history.messages)
    assert len(capped_prompt) <= 4096