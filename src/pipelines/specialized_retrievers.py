from chromadb import PersistentClient

from utils.weighted_hybrid import WeightedHybridRetriever


def create_specialized_retriever(domain: str) -> WeightedHybridRetriever:
    """
    Factory para criar retrievers especializados por domínio jurídico

    Args:
        domain: Domínio jurídico ('constitutional_law', 'consumer_law', 'human_rights_law')

    Returns:
        WeightedHybridRetriever configurado para o domínio específico

    Raises:
        ValueError: Se o domínio não for suportado ou a collection não existir
    """  # noqa: E501
    # Mapeamento de domínios para collections do Chroma
    domain_to_collection = {
        "constitutional_law": "constitutional_docs",
        "consumer_law": "consumer_docs",
        "human_rights_law": "human_rights_docs",
    }

    if domain not in domain_to_collection:
        msg = (
            f"Domínio jurídico não suportado: {domain}. "
            f"Domínios suportados: {list(domain_to_collection.keys())}"
        )
        raise ValueError(msg)

    collection_name = domain_to_collection[domain]

    try:
        # Conecta ao ChromaDB persistente
        client = PersistentClient(path="chroma_db")

        # Verifica se a collection existe
        collections = client.list_collections()
        collection_names = [col.name for col in collections]

        if collection_name not in collection_names:
            msg = (
                f"Collection '{collection_name}' não encontrada no ChromaDB. "
                f"Collections disponíveis: {collection_names}"
            )
            raise ValueError(  # noqa: TRY301
                msg
            )

        # Obtém a collection
        collection = client.get_collection(collection_name)

        # ✅ CORREÇÃO: Cria os retrievers necessários para o WeightedHybrid
        bm25_retriever = create_bm25_retriever(collection)
        vector_retriever = create_vector_retriever(collection)

        # Cria o retriever weighted hybrid especializado
        retriever = WeightedHybridRetriever(
            bm25_retriever=bm25_retriever,
            vector_retriever=vector_retriever,
            weight_bm25=0.4,  # Peso para BM25 (lexical)
            weight_vector=0.6,  # Peso para vector (semântico)
            top_k=5,  # Número de documentos a retornar
        )

        print(f"✅ Retriever especializado criado para: {domain} -> {collection_name}")
        return retriever  # noqa: TRY300

    except Exception as e:
        print(f"❌ Erro ao criar retriever para {domain}: {e}")
        raise


def create_bm25_retriever(collection):  # noqa: ANN001, ANN201
    """
    Cria um retriever BM25 para busca lexical
    """
    from langchain_classic.schema import Document  # noqa: PLC0415
    from langchain_community.retrievers import BM25Retriever  # noqa: PLC0415

    try:
        # Extrai documentos da collection para o BM25
        results = collection.peek(limit=100)
        documents = []

        if hasattr(results, "documents"):
            for i, doc_text in enumerate(results.documents):
                metadata = results.metadatas[i] if results.metadatas else {}
                documents.append(Document(page_content=doc_text, metadata=metadata))

        if documents:
            return BM25Retriever.from_documents(documents)
        # Fallback: retriever vazio
        return BM25Retriever.from_texts(["placeholder"])

    except Exception as e:  # noqa: BLE001
        print(f"⚠️ Erro ao criar BM25 retriever: {e}")
        return BM25Retriever.from_texts(["placeholder document"])


def create_vector_retriever(collection):  # noqa: ANN001, ANN201
    """
    Cria um retriever vetorial usando Chroma
    """
    # ✅ CORREÇÃO: Usa Chroma diretamente sem OpenAIEmbeddings explícito
    # Se você já tem embeddings no Chroma, podemos usar diretamente
    try:
        from langchain_community.vectorstores import Chroma  # noqa: PLC0415

        # ✅ Abordagem simplificada: usa Chroma como vector store
        # Assumindo que o Chroma já está configurado com embeddings
        vector_store = Chroma(
            collection_name=collection.name, persist_directory="chroma_db"
        )

        return vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        )

    except Exception as e:  # noqa: BLE001
        print(f"⚠️ Erro ao criar vector retriever: {e}")

        # ✅ Fallback alternativo: cria um Chroma retriever básico
        try:
            from dotenv import load_dotenv  # noqa: PLC0415
            from langchain_community.vectorstores import Chroma  # noqa: PLC0415
            from langchain_openai import OpenAIEmbeddings  # noqa: PLC0415

            load_dotenv()

            # Se precisar criar embeddings do zero
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small"
                # API key é pega automaticamente do ambiente
            )

            vector_store = Chroma(
                collection_name=collection.name,
                embedding_function=embeddings,
                persist_directory="chroma_db",
            )

            return vector_store.as_retriever(search_kwargs={"k": 5})

        except Exception as e2:  # noqa: BLE001
            print(f"❌ Erro no fallback do vector retriever: {e2}")
            # Último fallback: retriever básico
            from langchain_community.retrievers import (  # noqa: PLC0415
                VectorStoreRetriever,
            )

            return VectorStoreRetriever(vectorstore=None)


def get_available_domains() -> list:
    """Retorna lista de domínios jurídicos disponíveis"""
    return ["constitutional_law", "consumer_law", "human_rights_law"]


def check_chroma_connections() -> dict:
    """
    Verifica a conexão com todas as collections do Chroma
    """
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
