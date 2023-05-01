from langchain.schema import BaseMemory, BaseRetriever
from langchain.prompts.chat import BaseChatPromptTemplate
from typing import Dict, Any, List
import logging
class RetrievalChatMemory(BaseMemory):
    """A memory component that retrieves and loads relevant data.

    It does not actually save or clear any context.
    """

    prompt: BaseChatPromptTemplate  
    """The prompt template used to format messages."""

    retriever: BaseRetriever 
    """The retriever used to get relevant documents."""  

    input_key: str
    """The key for input data."""

    memory_key: str
    """The key for storing retrieved data in memory."""

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Loads relevant data into memory.

        Gets the first input variable from the prompt, uses the retriever to get 
        relevant documents based on the input data. Formats messages using the prompt 
        and page content of documents, stores first message for each doc in memory under memory_key.
        If exception, logs error and stores default "nothing" message.
        """
        prompt_key = self.prompt.input_variables[0]
        try: 
            docs = self.retriever.get_relevant_documents(inputs[self.input_key])
            return {self.memory_key: [self.prompt.format_messages(**{prompt_key:doc.page_content})[0] for doc in docs]}
        except Exception as e:
            logging.getLogger().error(e)
            return {self.memory_key: [self.prompt.format_messages(**{prompt_key:'nothing'})[0]]}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Does nothing to save context."""
        pass

    def memory_variables(self) -> List[str]:
        """Returns a list of just the memory key.""" 
        return [self.memory_key]  

    def clear(self) -> None: 
        """Does nothing to clear the memory."""
        pass