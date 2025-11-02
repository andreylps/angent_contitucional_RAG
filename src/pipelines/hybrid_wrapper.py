# src/pipelines/hybrid_wrapper.py


from langchain_classic.schema import BaseRetriever, Document

from pipelines.rag_hybrid import hybrid_search


class HybridRetriever(BaseRetriever):
    top_k: int = 5  # <-- campo declarado corretamente

    def _get_relevant_documents(self, query: str, **kwargs) -> list[Document]:  # noqa: ANN003, ARG002
        results = hybrid_search(query, top_k=self.top_k)
        return [Document(page_content=r) for r in results]

    async def _aget_relevant_documents(self, query: str, **kwargs) -> list[Document]:  # noqa: ANN003, ARG002
        results = hybrid_search(query, top_k=self.top_k)
        return [Document(page_content=r) for r in results]
