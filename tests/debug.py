#!/usr/bin/env python3
"""
Debug completo do ChromaDB - Descobre onde estão as collections
"""

import os
import sys
from pathlib import Path

# Adiciona o diretório src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))


def debug_chromadb():
    """Faz um debug completo do ChromaDB"""

    print("🔍 DEBUG COMPLETO DO CHROMADB")
    print("=" * 50)

    # 1. Verifica se o diretório chroma_db existe
    chroma_dir = Path("chroma_db")
    print(f"📁 Diretório chroma_db existe: {chroma_dir.exists()}")

    if chroma_dir.exists():
        print("📂 Conteúdo de chroma_db:")
        for item in chroma_dir.iterdir():
            print(f"   - {item.name} ({'dir' if item.is_dir() else 'file'})")

    # 2. Tenta conectar com ChromaDB de diferentes formas
    print("\n🔄 Tentando conectar com ChromaDB...")

    try:
        from chromadb import PersistentClient

        # Tentativa 1: Diretório padrão
        client1 = PersistentClient(path="chroma_db")
        collections1 = client1.list_collections()
        print(
            f"📊 Conexão padrão ('chroma_db') - Collections: {[c.name for c in collections1]}"
        )

    except Exception as e:
        print(f"❌ Erro na conexão padrão: {e}")

    try:
        # Tentativa 2: Diretório absoluto
        client2 = PersistentClient(path=str(Path("chroma_db").absolute()))
        collections2 = client2.list_collections()
        print(f"📊 Conexão absoluta - Collections: {[c.name for c in collections2]}")

    except Exception as e:
        print(f"❌ Erro na conexão absoluta: {e}")

    # 3. Verifica o código do specialized_retrievers
    print("\n📝 Verificando specialized_retrievers.py...")

    try:
        from pipelines.specialized_retrievers import list_collections

        collections_custom = list_collections()
        print(f"📊 list_collections() retornou: {collections_custom}")

    except Exception as e:
        print(f"❌ Erro em list_collections(): {e}")

    # 4. Verifica o document_processor
    print("\n🔧 Verificando document_processor...")

    try:
        from pipelines.document_processor import DocumentProcessor

        processor = DocumentProcessor()

        # Verifica qual path o processor está usando
        print(f"📁 DocumentProcessor path: {processor.vector_store._client._path}")

        # Tenta listar collections pelo processor
        collections_processor = processor.vector_store._client.list_collections()
        print(
            f"📊 Collections via DocumentProcessor: {[c.name for c in collections_processor]}"
        )

    except Exception as e:
        print(f"❌ Erro no DocumentProcessor: {e}")
        import traceback

        traceback.print_exc()


def check_data_directory():
    """Verifica se os documentos originais existem"""

    print("\n📂 VERIFICANDO DIRETÓRIO DATA...")
    data_dir = Path("data")

    if not data_dir.exists():
        print("❌ Diretório 'data' não encontrado!")
        return

    print("✅ Diretório 'data' encontrado")

    for subdir in ["constitution", "direitos_humanos-oea", "consumer_law"]:
        subdir_path = data_dir / subdir
        if subdir_path.exists():
            files = list(subdir_path.glob("*.*"))
            print(f"📁 {subdir}: {len(files)} arquivos")
            for f in files:
                print(f"   - {f.name}")
        else:
            print(f"❌ {subdir}: não encontrado")


if __name__ == "__main__":
    debug_chromadb()
    check_data_directory()

    print("\n🎯 PRÓXIMOS PASSOS:")
    print("1. Se collections aparecerem acima: problema de import/contexto")
    print("2. Se collections NÃO aparecerem: problema no processamento")
    print("3. Execute este script e me mostre o output completo!")
