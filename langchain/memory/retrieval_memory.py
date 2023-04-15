from langchain.schema import BaseMemory
from typing import Dict, Any

class RetrievalMemory(BaseMemory):
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return super().load_memory_variables(inputs)
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        return super().save_context(inputs, outputs)
    
    def clear(self) -> None:
        return super().clear()