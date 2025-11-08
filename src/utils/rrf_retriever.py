#!/usr/bin/env python3
"""
Reciprocal Rank Fusion (RRF) Retriever - MUITO MELHOR que weighted hybrid

RRF combina rankings de múltiplos retrievers de forma mais inteligente:
- Não depende de normalização de scores
- Trata cada retriever de forma justa
- Funciona melhor empiricamente que weighted average

Fórmula RRF: score(doc) = Σ(1 / (k + rank_i))
onde k é uma constante (geralmente 60) e rank_i é a posição do doc no retriever i
"""

import asyncio
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class RRFRetriever(BaseRetriever):
    """
    Retriever que usa Reciprocal Rank Fusion para combinar múltiplos retrievers

    ✅ VANTAGENS sobre Weighted Hybrid:
    - Scores de diferentes retrievers não precisam ser normalizados
    - Mais robusto a diferenças de scale entre retrievers
    - Empiricamente superior em diversos benchmarks
    - Remove duplicatas automaticamente
    """

    retrievers: list[Any]
    weights: list[float] | None = None
    k: int = 60  # Constante RRF (padrão: 60)
    top_k: int = 5

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,  # noqa: ARG002
    ) -> list[Document]:
        """Versão síncrona"""
        # Executa todos os retrievers
        all_results = []
        for retriever in self.retrievers:
            try:
                if hasattr(retriever, "invoke"):
                    docs = retriever.invoke(query)
                else:
                    docs = retriever.get_relevant_documents(query)
                all_results.append(docs)
            except Exception as e:  # noqa: BLE001
                print(f"⚠️ Retriever falhou: {e}")
                all_results.append([])

        # Aplica RRF
        return self._apply_rrf(all_results)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun | None = None,  # noqa: ARG002
    ) -> list[Document]:
        """Versão assíncrona - executa retrievers em paralelo"""
        tasks = []

        for retriever in self.retrievers:
            task = asyncio.create_task(self._ainvoke_retriever(retriever, query))
            tasks.append(task)

        all_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filtra exceptions
        valid_results = []
        for result in all_results:
            if isinstance(result, Exception):
                print(f"⚠️ Retriever async falhou: {result}")
                valid_results.append([])
            else:
                valid_results.append(result)

        return self._apply_rrf(valid_results)

    async def _ainvoke_retriever(self, retriever: Any, query: str) -> list[Document]:
        """Helper para executar retriever de forma assíncrona"""
        try:
            if hasattr(retriever, "ainvoke"):
                return await retriever.ainvoke(query)
            if hasattr(retriever, "_aget_relevant_documents"):
                return await retriever._aget_relevant_documents(query)  # noqa: SLF001

            # Fallback síncrono em thread separada
            loop = asyncio.get_event_loop()
            if hasattr(retriever, "invoke"):
                return await loop.run_in_executor(None, retriever.invoke, query)
            return await loop.run_in_executor(
                None, retriever.get_relevant_documents, query
            )
        except Exception as e:  # noqa: BLE001
            print(f"❌ Erro no retriever assíncrono: {e}")
            return []

    def _apply_rrf(self, all_results: list[list[Document]]) -> list[Document]:
        """
        Aplica Reciprocal Rank Fusion nos resultados

        RRF Formula: score(doc) = Σ(weight_i / (k + rank_i))

        Args:
            all_results: Lista de listas de documentos (um por retriever)

        Returns:
            Documentos ordenados por RRF score
        """
        # Dicionário para acumular scores: doc_id -> (doc, score)
        doc_scores: dict[str, tuple[Document, float]] = {}

        # Pesos padrão se não fornecidos
        weights = self.weights or [1.0] * len(all_results)

        # Para cada retriever
        for retriever_idx, docs in enumerate(all_results):
            weight = weights[retriever_idx]

            # Para cada documento retornado
            for rank, doc in enumerate(docs):
                # Cria ID único baseado no conteúdo
                doc_id = self._get_doc_id(doc)

                # Calcula RRF score: weight / (k + rank)
                # rank começa em 0, então rank+1 é a posição real
                rrf_score = weight / (self.k + rank + 1)

                # Se documento já existe, acumula score
                if doc_id in doc_scores:
                    current_doc, current_score = doc_scores[doc_id]
                    doc_scores[doc_id] = (current_doc, current_score + rrf_score)
                else:
                    # Primeiro retriever a retornar este doc
                    doc_scores[doc_id] = (doc, rrf_score)

        # Ordena por score decrescente
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)

        # Retorna top_k documentos
        return [doc for doc, score in sorted_docs[: self.top_k]]

    def _get_doc_id(self, doc: Document) -> str:
        """
        Gera ID único para documento

        Usa hash do conteúdo para detectar duplicatas
        """
        # Tenta usar ID dos metadados primeiro
        if "id" in doc.metadata:
            return str(doc.metadata["id"])

        # Usa chunk_id + source se disponível
        if "chunk_id" in doc.metadata and "source" in doc.metadata:
            return f"{doc.metadata['source']}_{doc.metadata['chunk_id']}"

        # Fallback: hash do conteúdo
        import hashlib  # noqa: PLC0415

        return hashlib.md5(doc.page_content.encode()).hexdigest()  # noqa: S324

    def get_rrf_scores(self, query: str) -> list[tuple[Document, float]]:
        """
        ✨ NOVO: Retorna documentos COM scores RRF (útil para debug)

        Args:
            query: Query de busca

        Returns:
            Lista de (documento, score_rrf)
        """
        all_results = []
        for retriever in self.retrievers:
            try:
                if hasattr(retriever, "invoke"):
                    docs = retriever.invoke(query)
                else:
                    docs = retriever.get_relevant_documents(query)
                all_results.append(docs)
            except Exception as e:  # noqa: BLE001
                print(f"⚠️ Retriever falhou: {e}")
                all_results.append([])

        # Aplica RRF e retorna com scores
        doc_scores: dict[str, tuple[Document, float]] = {}
        weights = self.weights or [1.0] * len(all_results)

        for retriever_idx, docs in enumerate(all_results):
            weight = weights[retriever_idx]

            for rank, doc in enumerate(docs):
                doc_id = self._get_doc_id(doc)
                rrf_score = weight / (self.k + rank + 1)

                if doc_id in doc_scores:
                    current_doc, current_score = doc_scores[doc_id]
                    doc_scores[doc_id] = (current_doc, current_score + rrf_score)
                else:
                    doc_scores[doc_id] = (doc, rrf_score)

        # Ordena e retorna top_k com scores
        sorted_with_scores = sorted(
            doc_scores.values(), key=lambda x: x[1], reverse=True
        )

        return sorted_with_scores[: self.top_k]


# ✨ EXEMPLO DE USO
if __name__ == "__main__":
    """
    Exemplo de como usar o RRF Retriever
    """
    print("🔍 Teste do RRF Retriever")
    print("=" * 50)

    # Simula dois retrievers com resultados diferentes
    from langchain_community.retrievers import BM25Retriever

    # Docs de exemplo
    docs = [
        Document(
            page_content="Artigo 5º da Constituição trata de direitos fundamentais"
        ),
        Document(page_content="O CDC protege o consumidor em relações de consumo"),
        Document(page_content="Garantias constitucionais são invioláveis"),
        Document(page_content="Direitos humanos na Convenção Americana"),
    ]

    # Cria retrievers
    bm25 = BM25Retriever.from_documents(docs, k=3)
    vector = BM25Retriever.from_documents(docs, k=3)  # Simulando vector

    # Cria RRF retriever
    rrf = RRFRetriever(retrievers=[bm25, vector], weights=[0.4, 0.6], k=60, top_k=3)

    # Testa
    query = "direitos fundamentais"
    results = rrf.invoke(query)

    print(f"\n📊 Query: '{query}'")
    print(f"✅ Resultados: {len(results)} documentos")

    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc.page_content[:100]}...")
