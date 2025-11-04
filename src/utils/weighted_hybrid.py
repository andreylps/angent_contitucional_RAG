# pipelines/weighted_hybrid.py

import asyncio
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class WeightedHybridRetriever(BaseRetriever):
    """
    Retriever híbrido ponderado entre BM25 (lexical) e Chroma (vetorial).
    Compatível com LangChain 0.2+
    """

    bm25_retriever: Any
    vector_retriever: Any
    weight_bm25: float = 0.4
    weight_vector: float = 0.6
    top_k: int = 5

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,  # noqa: ARG002
    ) -> list[Document]:
        """
        Versão síncrona usada internamente pela chain.
        """
        # Obter documentos do BM25
        if hasattr(self.bm25_retriever, "invoke"):
            bm25_docs = self.bm25_retriever.invoke(query)
        else:
            bm25_docs = self.bm25_retriever.get_relevant_documents(query)

        # Obter documentos do vetorial
        if hasattr(self.vector_retriever, "invoke"):
            vector_docs = self.vector_retriever.invoke(query)
        else:
            vector_docs = self.vector_retriever.get_relevant_documents(query)

        # Combina os resultados ponderando os scores
        combined = []
        num_bm25 = int(self.top_k * self.weight_bm25)
        num_vector = self.top_k - num_bm25

        combined.extend(bm25_docs[:num_bm25])
        combined.extend(vector_docs[:num_vector])

        return combined

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun | None = None,  # noqa: ARG002
    ) -> list[Document]:
        """
        Versão assíncrona compatível com LangChain.
        """
        # Executar ambas as buscas em paralelo
        bm25_task = asyncio.create_task(
            self._ainvoke_retriever(self.bm25_retriever, query)
        )
        vector_task = asyncio.create_task(
            self._ainvoke_retriever(self.vector_retriever, query)
        )

        bm25_docs, vector_docs = await asyncio.gather(bm25_task, vector_task)

        # Combina os resultados
        combined = []
        num_bm25 = int(self.top_k * self.weight_bm25)
        num_vector = self.top_k - num_bm25

        combined.extend(bm25_docs[:num_bm25])
        combined.extend(vector_docs[:num_vector])

        return combined

    async def _ainvoke_retriever(self, retriever: Any, query: str) -> list[Document]:
        """Helper method para chamadas assíncronas"""
        if hasattr(retriever, "ainvoke"):
            return await retriever.ainvoke(query)
        if hasattr(retriever, "_aget_relevant_documents"):
            return await retriever._aget_relevant_documents(query)  # noqa: SLF001
        # Fallback para síncrono em thread separada
        loop = asyncio.get_event_loop()
        if hasattr(retriever, "invoke"):
            return await loop.run_in_executor(None, retriever.invoke, query)
        return await loop.run_in_executor(None, retriever.get_relevant_documents, query)
