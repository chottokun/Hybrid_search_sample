from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List, Optional, Any

class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever: BaseRetriever, bm25_retriever: Optional[BaseRetriever] = None, weights: Optional[List[float]] = None):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.weights = weights if weights is not None else [0.5, 0.5]

    def get_relevant_documents(self, query: str, *, callbacks=None, tags=None, metadata=None, run_name=None, **kwargs: Any) -> List[Document]:
        vector_docs = self.vector_retriever.get_relevant_documents(query, callbacks=callbacks, tags=tags, metadata=metadata, run_name=run_name, **kwargs) if self.vector_retriever else []
        bm25_docs = self.bm25_retriever.get_relevant_documents(query, callbacks=callbacks, tags=tags, metadata=metadata, run_name=run_name, **kwargs) if self.bm25_retriever else []
        return self._merge_results(vector_docs, bm25_docs)

    def _merge_results(self, vector_docs: List[Document], bm25_docs: List[Document]) -> List[Document]:
        seen = set()
        merged = []
        for doc in vector_docs + bm25_docs:
            if doc.page_content not in seen:
                merged.append(doc)
                seen.add(doc.page_content)
        return merged

    async def aget_relevant_documents(self, query: str, *, callbacks=None, tags=None, metadata=None, run_name=None, **kwargs: Any) -> List[Document]:
        vector_docs = await self.vector_retriever.aget_relevant_documents(query, callbacks=callbacks, tags=tags, metadata=metadata, run_name=run_name, **kwargs) if self.vector_retriever else []
        bm25_docs = await self.bm25_retriever.aget_relevant_documents(query, callbacks=callbacks, tags=tags, metadata=metadata, run_name=run_name, **kwargs) if self.bm25_retriever else []
        return self._merge_results(vector_docs, bm25_docs)
