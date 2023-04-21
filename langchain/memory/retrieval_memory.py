from langchain.schema import BaseMemory, BaseRetriever
from typing import Dict, Any, List
import logging

class RetrievalChatMemory(BaseMemory):
    
    prompt: Any
    retriever: BaseRetriever
    input_key: str
    prompt_key: str
    memory_key: str
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        try:
            docs = self.retriever.get_relevant_documents(inputs[self.input_key])
            return {self.memory_key: [self.prompt.format_messages(**{self.prompt_key:doc.page_content})[0] for doc in docs]}
        except Exception as e:
            logging.getLogger().error(e)
            return {self.memory_key: [self.prompt.format_messages(**{self.prompt_key:'nothing'})[0]]}
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        # nothing to do
        pass
    
    def memory_variables(self) -> List[str]:
        return [self.memory_key]
    
    def clear(self) -> None:
        pass