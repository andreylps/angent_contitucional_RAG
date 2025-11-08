#!/usr/bin/env python3
"""
Retrievers especializados OTIMIZADOS com RRF e correções críticas
"""

from typing import Any

from chromadb import PersistentClient
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


def create_specialized_retriever(domain: str, use_rrf: bool = True) -> Any:  # noqa: FBT001, FBT002
    """
    Factory para criar retrievers especializados OTIMIZADOS

    Args:
        domain: Domínio jurídico ('constitutional_law', 'consumer_law', 'human_rights_law')
        use_rrf: Se True, usa Reciprocal Rank Fusion (recomendado)

    Returns:
        Retriever otimizado com RRF ou Weighted Hybrid
    """  # noqa: E501
    # Mapeamento de domínios
    domain_to_collection = {
        "constitutional_law": "constitutional_docs",
        "consumer_law": "consumer_docs",
        "human_rights_law": "human_rights_docs",
    }

    if domain not in domain_to_collection:
        msg = f"Domínio não suportado: {domain}. Suportados: {list(domain_to_collection.keys())}"  # noqa: E501
        raise ValueError(msg)

    collection_name = domain_to_collection[domain]

    try:
        # Conecta ao ChromaDB
        client = PersistentClient(path="chroma_db")

        # Verifica se collection existe
        collections = client.list_collections()
        collection_names = [col.name for col in collections]

        if collection_name not in collection_names:
            msg = f"Collection '{collection_name}' não encontrada. Disponíveis: {collection_names}"  # noqa: E501
            raise ValueError(msg)  # noqa: TRY301

        collection = client.get_collection(collection_name)

        print(f"✅ Criando retriever otimizado para: {domain}")
        print(f"   📚 Collection: {collection_name} ({collection.count()} documentos)")

        # ✅ CORREÇÃO CRÍTICA: BM25 com TODOS os documentos
        bm25_retriever = create_bm25_retriever_full(collection)

        # ✅ Vector retriever otimizado
        vector_retriever = create_vector_retriever_optimized(collection)

        if use_rrf:
            # ✅ NOVO: Usa Reciprocal Rank Fusion
            from ..utils.rrf_retriever import RRFRetriever  # noqa: PLC0415, TID252

            retriever = RRFRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[0.4, 0.6],  # BM25: 40%, Vector: 60%
                k=60,  # Constante RRF
                top_k=5,
            )
            print("   🎯 Modo: Reciprocal Rank Fusion (RRF)")
        else:
            # Fallback: Weighted Hybrid original
            from ..utils.weighted_hybrid import (  # noqa: PLC0415, TID252
                WeightedHybridRetriever,
            )

            retriever = WeightedHybridRetriever(
                bm25_retriever=bm25_retriever,
                vector_retriever=vector_retriever,
                weight_bm25=0.4,
                weight_vector=0.6,
                top_k=5,
            )
            print("   ⚖️ Modo: Weighted Hybrid")

        return retriever  # noqa: TRY300

    except Exception as e:
        print(f"❌ Erro ao criar retriever: {e}")
        raise


def create_bm25_retriever_full(collection: Any) -> BM25Retriever:
    """
    ✅ CORREÇÃO CRÍTICA: BM25 com TODOS os documentos

    ANTES: Usava apenas 100 documentos (peek limit=100)
    AGORA: Indexa TODOS os documentos para IDF correto
    """
    try:
        # ✅ CORREÇÃO FINAL: Substitui a lógica quebrada e duplicada por uma única chamada limpa.
        # O método .get() do ChromaDB, sem limit, busca todos os documentos da coleção.
        all_docs_data = collection.get(include=["documents", "metadatas"])
        all_documents = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(all_docs_data["documents"], all_docs_data["metadatas"])
        ]
        print(f"   📊 Indexando {len(all_documents)} documentos no BM25...")

        print(f"   ✅ BM25 indexado com {len(all_documents)} documentos")

        if all_documents:
            return BM25Retriever.from_documents(
                all_documents,
                k=10,
            )

        # Fallback
        print("   ⚠️ Nenhum documento encontrado, usando placeholder")
        return BM25Retriever.from_texts(["placeholder"], k=5)

    except Exception as e:  # noqa: BLE001
        print(f"   ❌ Erro ao criar BM25: {e}")
        return BM25Retriever.from_texts(["placeholder"], k=5)


def create_vector_retriever_optimized(collection: Any) -> Any:
    """
    ✅ Vector retriever otimizado com configurações melhores
    """
    try:
        # ✅ OTIMIZAÇÃO: Usa o mesmo modelo de embedding local da ingestão.
        # ✅ ATUALIZAÇÃO: Usa o novo pacote modular da LangChain.
        from langchain_community.embeddings import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )

        # Vector store com collection existente
        vector_store = Chroma(
            collection_name=collection.name,
            embedding_function=embeddings,
            persist_directory="chroma_db",
        )

        # ✅ Configuração otimizada do retriever
        return vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 10,  # Retorna top 10 para RRF
            },
        )

    except Exception as e:
        print(f"   ❌ Erro ao criar vector retriever: {e}")
        raise


def get_available_domains() -> list[str]:
    """Retorna lista de domínios jurídicos disponíveis"""
    return ["constitutional_law", "consumer_law", "human_rights_law"]


def check_chroma_connections() -> dict[str, Any]:
    """Verifica conexão com ChromaDB e retorna status"""
    try:
        client = PersistentClient(path="chroma_db")
        collections = client.list_collections()

        status = {}
        for collection in collections:
            status[collection.name] = {
                "count": collection.count(),
                "metadata": collection.metadata or {},
            }

        return {
            "status": "connected",
            "collections": status,
            "total_collections": len(collections),
        }

    except Exception as e:  # noqa: BLE001
        return {
            "status": "error",
            "error": str(e),
            "collections": {},
            "total_collections": 0,
        }


def benchmark_retriever(retriever: Any, test_queries: list[str]) -> dict[str, Any]:
    """
    ✨ NOVO: Benchmark de performance do retriever

    Args:
        retriever: Retriever a ser testado
        test_queries: Lista de queries de teste

    Returns:
        Métricas de performance
    """
    import time  # noqa: PLC0415

    results = {
        "total_queries": len(test_queries),
        "avg_latency": 0.0,
        "avg_docs_returned": 0.0,
        "queries_results": [],
    }

    latencies = []
    docs_counts = []

    for query in test_queries:
        start = time.time()

        try:
            docs = retriever.invoke(query)
            latency = time.time() - start

            latencies.append(latency)
            docs_counts.append(len(docs))

            results["queries_results"].append(
                {
                    "query": query,
                    "latency": latency,
                    "docs_found": len(docs),
                    "status": "success",
                }
            )

        except Exception as e:  # noqa: BLE001
            results["queries_results"].append(
                {
                    "query": query,
                    "latency": 0,
                    "docs_found": 0,
                    "status": "error",
                    "error": str(e),
                }
            )

    if latencies:
        results["avg_latency"] = sum(latencies) / len(latencies)
        results["avg_docs_returned"] = sum(docs_counts) / len(docs_counts)

    return results
