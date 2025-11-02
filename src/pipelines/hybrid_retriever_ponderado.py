import numpy as np
from langchain_classic.schema import BaseRetriever, Document
from rank_bm25 import BM25Okapi


class WeightedHybridRetriever(BaseRetriever):
    def __init__(
        self,
        bm25_docs: list[str],
        vector_retriever,  # noqa: ANN001
        top_k: int = 5,
        weight_bm25: float = 0.5,
        weight_vector: float = 0.5,
    ) -> None:
        self.top_k = top_k
        self.vector_retriever = vector_retriever
        self.weight_bm25 = weight_bm25
        self.weight_vector = weight_vector

        # Inicializa BM25
        tokenized_docs = [doc.split() for doc in bm25_docs]
        self.bm25 = BM25Okapi(tokenized_docs)
        self.bm25_docs = bm25_docs

    def _get_relevant_documents(self, query: str) -> list[Document]:  # pyright: ignore[reportIncompatibleMethodOverride]
        # BM25
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        # Normaliza BM25
        bm25_scores = (bm25_scores - bm25_scores.min()) / (
            bm25_scores.max() - bm25_scores.min() + 1e-8
        )

        # Vetorial
        vector_docs = self.vector_retriever.get_relevant_documents(query)
        # Extrai scores dos vetoriais (assumindo que o retriever retorna score em doc.metadata['score'])  # noqa: E501
        vector_scores = np.array(
            [doc.metadata.get("score", 1.0) for doc in vector_docs]
        )
        vector_scores = (vector_scores - vector_scores.min()) / (
            vector_scores.max() - vector_scores.min() + 1e-8
        )

        # Combina documentos e scores
        combined_docs = []
        for idx, score in enumerate(bm25_scores):
            combined_docs.append(
                {
                    "doc": Document(page_content=self.bm25_docs[idx]),
                    "score": score * self.weight_bm25,
                }
            )

        for idx, doc in enumerate(vector_docs):
            combined_docs.append(
                {"doc": doc, "score": vector_scores[idx] * self.weight_vector}
            )

        # Ordena pelo score combinado
        combined_docs = sorted(combined_docs, key=lambda x: x["score"], reverse=True)

        # Retorna apenas os Document objects
        return [item["doc"] for item in combined_docs[: self.top_k]]

    async def _aget_relevant_documents(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        query: str,
        run_manager=None,  # noqa: ANN001, ARG002
    ) -> list[Document]:
        return self._get_relevant_documents(query)
