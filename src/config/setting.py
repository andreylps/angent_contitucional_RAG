#!/usr/bin/env python3
"""
Configurações centralizadas do sistema RAG Jurídico

Facilita ajustes de performance, modelos e parâmetros
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from pydantic import SecretStr

# Carrega .env
load_dotenv()


@dataclass
class LLMConfig:
    """Configurações de LLM"""

    # Modelo principal
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 2000

    # Modelo para tarefas específicas
    router_model: str = "gpt-4o-mini"  # Pode usar modelo mais barato
    summary_model: str = "gpt-4o-mini"

    # API
    api_key: SecretStr = os.getenv("OPENAI_API_KEY", "")  # type: ignore  # noqa: PGH003, RUF009

    # Rate limiting
    max_retries: int = 3
    timeout: int = 60


@dataclass
class EmbeddingsConfig:
    """Configurações de embeddings"""

    # Modelo
    model: str = "text-embedding-3-small"  # Rápido e barato
    # model: str = "text-embedding-3-large"  # Mais preciso mas caro

    # Dimensões (3-small: 1536, 3-large: 3072)
    dimensions: int = 1536

    # Batch size para processar embeddings
    chunk_size: int = 1000

    # Cache
    use_cache: bool = True


@dataclass
class ChunkingConfig:
    """Configurações de chunking de documentos"""

    # Tamanho dos chunks (em caracteres)
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Separadores hierárquicos
    separators: list[str] | None = None

    def __post_init__(self):  # noqa: ANN204
        if self.separators is None:
            # Separadores jurídicos específicos
            self.separators = [
                "\n\n## ",  # Seções
                "\n\nArt. ",  # Artigos
                "\n\n",  # Parágrafos
                "\n",  # Linhas
                " ",  # Palavras
                "",  # Caracteres
            ]

    # Metadata enrichment
    add_section_metadata: bool = True
    extract_article_numbers: bool = True


@dataclass
class RetrievalConfig:
    """Configurações de retrieval"""

    # Número de documentos a retornar
    top_k: int = 5

    # Retrieval híbrido
    use_hybrid: bool = True
    bm25_weight: float = 0.4
    vector_weight: float = 0.6

    # RRF (Reciprocal Rank Fusion)
    use_rrf: bool = True  # Recomendado!
    rrf_k: int = 60  # Constante RRF

    # Fetch mais documentos internamente para re-ranking
    fetch_k: int = 20

    # Re-ranking
    use_reranking: bool = False  # Ativa quando tiver cross-encoder
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 3


@dataclass
class RouterConfig:
    """Configurações do router agent"""

    # Modo de roteamento
    mode: str = "semantic"  # "keyword" ou "semantic"

    # Threshold para seleção de agentes
    confidence_threshold: float = 0.3

    # Número máximo de agentes a ativar simultaneamente
    max_agents: int = 2

    # Fallback para todos os agentes se confiança baixa
    fallback_to_all: bool = False


@dataclass
class CacheConfig:
    """Configurações de cache"""

    # Ativar cache
    enabled: bool = True

    # Diretório
    cache_dir: str = "cache"

    # TTL (Time To Live) em segundos
    ttl_seconds: int = 3600 * 24 * 7  # 7 dias

    # Tamanho máximo em MB
    max_size_mb: int = 500

    # Cache por tipo
    cache_embeddings: bool = True
    cache_queries: bool = True
    cache_responses: bool = True


@dataclass
class ChromaConfig:
    """Configurações do ChromaDB"""

    # Diretório
    persist_directory: str = "chroma_db"

    # Collections
    collections: dict[str, str] | None = None

    def __post_init__(self):  # noqa: ANN204
        if self.collections is None:
            self.collections = {
                "constitutional_law": "constitutional_docs",
                "consumer_law": "consumer_docs",
                "human_rights_law": "human_rights_docs",
            }


@dataclass
class SystemConfig:
    """Configuração geral do sistema"""

    # Debug mode
    debug: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"
    verbose: bool = False

    # Logs
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    log_dir: str = "logs"

    # Performance
    enable_async: bool = True
    max_concurrent_agents: int = 3

    # Streaming
    enable_streaming: bool = False  # Para implementar depois


class Settings:
    """
    Configurações centralizadas do sistema

    Uso:
        from config.settings import settings

        # Acessa configs
        model = settings.llm.model
        top_k = settings.retrieval.top_k
    """

    def __init__(self) -> None:
        self.llm = LLMConfig()
        self.embeddings = EmbeddingsConfig()
        self.chunking = ChunkingConfig()
        self.retrieval = RetrievalConfig()
        self.router = RouterConfig()
        self.cache = CacheConfig()
        self.chroma = ChromaConfig()
        self.system = SystemConfig()

    def validate(self) -> bool:
        """Valida configurações"""
        errors = []

        # Verifica API key
        if not self.llm.api_key:
            errors.append("OPENAI_API_KEY não encontrada no .env")

        # Verifica pesos do hybrid
        if self.retrieval.use_hybrid:
            total_weight = self.retrieval.bm25_weight + self.retrieval.vector_weight
            if abs(total_weight - 1.0) > 0.01:  # noqa: PLR2004
                errors.append(f"Pesos do hybrid devem somar 1.0, não {total_weight}")

        # Verifica diretórios
        required_dirs = [
            self.cache.cache_dir,
            self.system.log_dir,
            self.chroma.persist_directory,
        ]

        for dir_path in required_dirs:
            Path(dir_path).mkdir(exist_ok=True, parents=True)

        if errors:
            print("❌ Erros de configuração:")
            for error in errors:
                print(f"   - {error}")
            return False

        print("✅ Configurações validadas com sucesso")
        return True

    def print_summary(self) -> None:
        """Imprime resumo das configurações"""
        print("\n" + "=" * 60)
        print("⚙️  CONFIGURAÇÕES DO SISTEMA RAG JURÍDICO")
        print("=" * 60)

        print("\n🤖 LLM:")
        print(f"   Modelo: {self.llm.model}")
        print(f"   Temperature: {self.llm.temperature}")
        print(f"   Max Tokens: {self.llm.max_tokens}")

        print("\n📊 Embeddings:")
        print(f"   Modelo: {self.embeddings.model}")
        print(f"   Dimensões: {self.embeddings.dimensions}")
        print(f"   Cache: {'✅' if self.embeddings.use_cache else '❌'}")

        print("\n✂️  Chunking:")
        print(f"   Chunk Size: {self.chunking.chunk_size}")
        print(f"   Overlap: {self.chunking.chunk_overlap}")

        print("\n🔍 Retrieval:")
        print(f"   Top K: {self.retrieval.top_k}")
        print(f"   Modo: {'RRF' if self.retrieval.use_rrf else 'Weighted Hybrid'}")
        if self.retrieval.use_hybrid:
            print(
                f"   Pesos: BM25={self.retrieval.bm25_weight}, Vector={self.retrieval.vector_weight}"  # noqa: E501
            )
        print(f"   Re-ranking: {'✅' if self.retrieval.use_reranking else '❌'}")

        print("\n🎯 Router:")
        print(f"   Modo: {self.router.mode}")
        print(f"   Max Agentes: {self.router.max_agents}")

        print("\n💾 Cache:")
        print(f"   Ativo: {'✅' if self.cache.enabled else '❌'}")
        print(f"   TTL: {self.cache.ttl_seconds / 3600:.0f}h")
        print(f"   Max Size: {self.cache.max_size_mb}MB")

        print("\n🗄️  ChromaDB:")
        print(f"   Diretório: {self.chroma.persist_directory}")
        print(
            f"   Collections: {len(self.chroma.collections) if self.chroma.collections else 0}"  # noqa: E501
        )

        print("\n🔧 Sistema:")
        print(f"   Debug: {'✅' if self.system.debug else '❌'}")
        print(f"   Async: {'✅' if self.system.enable_async else '❌'}")
        print(f"   Log Level: {self.system.log_level}")

        print("=" * 60 + "\n")


# ✨ Singleton global
settings = Settings()


# ✨ EXEMPLO DE USO
if __name__ == "__main__":
    print("⚙️  Teste das Configurações")

    # Valida
    settings.validate()

    # Mostra resumo
    settings.print_summary()

    # Acessa valores específicos
    print("\n📝 Exemplos de uso:")
    print(f"   LLM Model: {settings.llm.model}")
    print(f"   Top K: {settings.retrieval.top_k}")
    print(f"   Cache ativo: {settings.cache.enabled}")

    # Modifica valores
    settings.retrieval.top_k = 10
    print(f"\n✅ Top K modificado: {settings.retrieval.top_k}")
