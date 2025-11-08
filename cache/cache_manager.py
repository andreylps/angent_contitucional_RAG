#!/usr/bin/env python3
"""
Sistema de cache para otimizar performance do RAG

Cacheia:
- Embeddings de queries (evita chamadas à API OpenAI)
- Resultados de retrieval (queries repetidas)
- Respostas de agentes (para perguntas idênticas)

Performance esperada:
- Queries cacheadas: 10-50x mais rápidas
- Economia de API calls: 60-80% em produção
"""

import hashlib
import os
import pickle
import time
from pathlib import Path
from typing import Any


class CacheManager:
    """
    Gerenciador de cache simples e eficiente

    Usa sistema de arquivos (disk cache) para persistência
    """

    def __init__(
        self,
        cache_dir: str = "cache",
        ttl_seconds: int = 3600 * 24 * 7,  # 7 dias padrão
        max_cache_size_mb: int = 500,  # 500MB máximo
    ) -> None:
        """
        Args:
            cache_dir: Diretório para armazenar cache
            ttl_seconds: Time-to-live em segundos (padrão: 7 dias)
            max_cache_size_mb: Tamanho máximo do cache em MB
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        self.ttl_seconds = ttl_seconds
        self.max_cache_size_bytes = max_cache_size_mb * 1024 * 1024

        # Subdiretórios por tipo de cache
        self.embeddings_dir = self.cache_dir / "embeddings"
        self.queries_dir = self.cache_dir / "queries"
        self.responses_dir = self.cache_dir / "responses"

        for dir_path in [self.embeddings_dir, self.queries_dir, self.responses_dir]:
            dir_path.mkdir(exist_ok=True)

        # Stats
        self.stats = {
            "hits": 0,
            "misses": 0,
            "size_bytes": 0,
        }

        print(f"✅ CacheManager inicializado: {self.cache_dir}")
        self._update_stats()

    def get_embedding(self, text: str) -> list[float] | None:
        """
        Busca embedding cacheado

        Args:
            text: Texto para buscar embedding

        Returns:
            Embedding (lista de floats) ou None se não encontrado
        """
        cache_key = self._get_cache_key(text)
        cache_path = self.embeddings_dir / f"{cache_key}.pkl"

        return self._get_from_cache(cache_path)

    def set_embedding(self, text: str, embedding: list[float]) -> None:
        """
        Cacheia embedding

        Args:
            text: Texto original
            embedding: Vetor de embedding
        """
        cache_key = self._get_cache_key(text)
        cache_path = self.embeddings_dir / f"{cache_key}.pkl"

        self._set_to_cache(cache_path, embedding)

    def get_query_results(
        self, query: str, retriever_id: str = "default"
    ) -> list[Any] | None:
        """
        Busca resultados de retrieval cacheados

        Args:
            query: Query de busca
            retriever_id: ID do retriever (para diferenciar retrievers)

        Returns:
            Lista de documentos ou None
        """
        cache_key = self._get_cache_key(f"{retriever_id}_{query}")
        cache_path = self.queries_dir / f"{cache_key}.pkl"

        return self._get_from_cache(cache_path)

    def set_query_results(
        self, query: str, results: list[Any], retriever_id: str = "default"
    ) -> None:
        """
        Cacheia resultados de retrieval

        Args:
            query: Query de busca
            results: Lista de documentos
            retriever_id: ID do retriever
        """
        cache_key = self._get_cache_key(f"{retriever_id}_{query}")
        cache_path = self.queries_dir / f"{cache_key}.pkl"

        self._set_to_cache(cache_path, results)

    def get_agent_response(self, query: str, agent_id: str) -> dict[str, Any] | None:
        """
        Busca resposta de agente cacheada

        Args:
            query: Query original
            agent_id: ID do agente

        Returns:
            Dict com resposta ou None
        """
        cache_key = self._get_cache_key(f"{agent_id}_{query}")
        cache_path = self.responses_dir / f"{cache_key}.pkl"

        return self._get_from_cache(cache_path)

    def set_agent_response(
        self, query: str, agent_id: str, response: dict[str, Any]
    ) -> None:
        """
        Cacheia resposta de agente

        Args:
            query: Query original
            agent_id: ID do agente
            response: Resposta completa do agente
        """
        cache_key = self._get_cache_key(f"{agent_id}_{query}")
        cache_path = self.responses_dir / f"{cache_key}.pkl"

        self._set_to_cache(cache_path, response)

    def _get_cache_key(self, text: str) -> str:
        """Gera chave de cache (hash MD5)"""
        return hashlib.md5(text.encode()).hexdigest()

    def _get_from_cache(self, cache_path: Path) -> Any | None:
        """
        Busca item do cache

        Verifica TTL e validade
        """
        if not cache_path.exists():
            self.stats["misses"] += 1
            return None

        try:
            # Verifica TTL
            age = time.time() - cache_path.stat().st_mtime
            if age > self.ttl_seconds:
                # Cache expirado
                cache_path.unlink()
                self.stats["misses"] += 1
                return None

            # Lê do cache
            with open(cache_path, "rb") as f:
                data = pickle.load(f)

            self.stats["hits"] += 1
            return data

        except Exception as e:
            print(f"⚠️ Erro ao ler cache: {e}")
            self.stats["misses"] += 1
            return None

    def _set_to_cache(self, cache_path: Path, data: Any) -> None:
        """
        Salva item no cache

        Gerencia tamanho máximo do cache
        """
        try:
            # Verifica tamanho do cache
            self._check_cache_size()

            # Salva
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)

            self._update_stats()

        except Exception as e:
            print(f"⚠️ Erro ao salvar cache: {e}")

    def _check_cache_size(self) -> None:
        """
        Verifica e limpa cache se necessário

        Remove arquivos mais antigos se exceder limite
        """
        current_size = self._get_cache_size()

        if current_size > self.max_cache_size_bytes:
            print(f"⚠️ Cache excedeu limite ({current_size / 1024 / 1024:.1f}MB)")
            self._cleanup_old_files()

    def _get_cache_size(self) -> int:
        """Calcula tamanho total do cache em bytes"""
        total_size = 0
        for root, dirs, files in os.walk(self.cache_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.exists(file_path):  # ✅ CORREÇÃO
                    total_size += os.path.getsize(file_path)
        return total_size

    def _cleanup_old_files(self, keep_newest: int = 1000) -> None:
        """
        Remove arquivos mais antigos do cache

        Args:
            keep_newest: Número de arquivos mais recentes a manter
        """
        all_files = []

        for root, dirs, files in os.walk(self.cache_dir):
            for file in files:
                file_path = Path(root) / file
                if file_path.exists():  # ✅ CORREÇÃO
                    all_files.append((file_path, file_path.stat().st_mtime))

        # Ordena por data (mais recentes primeiro)
        all_files.sort(key=lambda x: x[1], reverse=True)

        # Remove arquivos antigos
        removed = 0
        for file_path, _ in all_files[keep_newest:]:
            try:
                if file_path.exists():  # ✅ CORREÇÃO
                    file_path.unlink()
                    removed += 1
            except Exception as e:
                print(f"⚠️ Erro ao remover {file_path}: {e}")

        if removed > 0:
            print(f"🧹 Cache limpo: {removed} arquivos removidos")
        self._update_stats()

    def _update_stats(self) -> None:
        """Atualiza estatísticas do cache"""
        self.stats["size_bytes"] = self._get_cache_size()

    def get_stats(self) -> dict[str, Any]:
        """
        Retorna estatísticas do cache

        Returns:
            Dict com métricas de uso
        """
        self._update_stats()

        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0

        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": f"{hit_rate * 100:.1f}%",
            "size_mb": f"{self.stats['size_bytes'] / 1024 / 1024:.1f}",
            "ttl_hours": self.ttl_seconds / 3600,
        }

    def clear_all(self) -> None:
        """Limpa todo o cache"""
        import shutil

        for subdir in [self.embeddings_dir, self.queries_dir, self.responses_dir]:
            if subdir.exists():
                shutil.rmtree(subdir)
                subdir.mkdir(exist_ok=True)

        self.stats = {"hits": 0, "misses": 0, "size_bytes": 0}
        print("🧹 Cache completamente limpo")

    def clear_expired(self) -> None:
        """Remove apenas itens expirados"""
        removed = 0

        for root, dirs, files in os.walk(self.cache_dir):
            for file in files:
                file_path = Path(root) / file
                if not file_path.exists():  # ✅ CORREÇÃO
                    continue

                age = time.time() - file_path.stat().st_mtime

                if age > self.ttl_seconds:
                    try:
                        file_path.unlink()
                        removed += 1
                    except Exception as e:
                        print(f"⚠️ Erro ao remover {file_path}: {e}")

        if removed > 0:
            print(f"🧹 {removed} arquivos expirados removidos")
        self._update_stats()


# ✨ Singleton global
_cache_instance: CacheManager | None = None


def get_cache_manager() -> CacheManager:
    """Retorna instância singleton do CacheManager"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = CacheManager()
    return _cache_instance


# ✨ EXEMPLO DE USO
if __name__ == "__main__":
    print("🧪 Teste do Cache Manager")
    print("=" * 50)

    cache = CacheManager(cache_dir="test_cache", ttl_seconds=60)

    # Testa cache de embedding
    print("\n1️⃣ Testando cache de embeddings...")
    text = "Quais são os direitos fundamentais?"
    embedding = [0.1, 0.2, 0.3] * 100  # Simula embedding

    cache.set_embedding(text, embedding)
    cached = cache.get_embedding(text)
    if cached:
        print(f"✅ Embedding cacheado: {cached[:3]}... (hit!)")

    # Testa cache de query
    print("\n2️⃣ Testando cache de queries...")
    query = "CDC consumidor"
    results = ["doc1", "doc2", "doc3"]

    cache.set_query_results(query, results)
    cached_results = cache.get_query_results(query)
    if cached_results:
        print(f"✅ Resultados cacheados: {cached_results} (hit!)")

    # Stats
    print("\n📊 Estatísticas:")
    stats = cache.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # Cleanup
    import shutil

    if Path("test_cache").exists():
        shutil.rmtree("test_cache")
        print("\n🧹 Cache de teste removido")
