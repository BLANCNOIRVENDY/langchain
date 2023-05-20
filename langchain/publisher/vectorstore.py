from typing import List
from langchain.schema import BasePublisher, Document
from langchain.vectorstores.base import VectorStore

class VectorStorePublisher(BasePublisher):
    vectorstore: VectorStore
    def __init__(self, vectorstore: VectorStore) -> None:
        super().__init__()
        self.vectorstore = vectorstore
    
    def publish(self, docs: List[Document]):
        self.vectorstore.add_documents(docs)
    
    async def apublish(self, docs: List[Document]):
        await self.vectorstore.aadd_documents(docs)