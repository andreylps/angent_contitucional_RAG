#!/usr/bin/env python3
"""
Script para processar documentos jurídicos e criar collections no Chroma
"""

import os
import shutil
import sys

# Adiciona o src ao path para imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # noqa: PTH118, PTH120

from src.pipelines.document_processor import DocumentProcessor
from src.pipelines.specialized_retrievers import (
    check_chroma_connections,
)
from src.utils.logger import logger


def cleanup_database(db_path: str = "chroma_db") -> None:
    """Apaga o diretório do banco de dados Chroma para garantir uma nova criação."""
    if os.path.exists(db_path):  # noqa: PTH110
        logger.info(f"🧹 Limpando banco de dados antigo em '{db_path}'...")
        try:
            shutil.rmtree(db_path)
            logger.info("   ✅ Banco de dados antigo removido com sucesso.")
        except Exception as e:  # noqa: BLE001
            logger.error(f"   ❌ Erro ao remover o banco de dados: {e}")
            sys.exit(1)  # Aborta o script se a limpeza falhar


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """Processa todos os documentos jurídicos"""
    # ✅ ETAPA 1: Limpa o banco de dados antigo
    cleanup_database()

    logger.info("\n📚 PROCESSANDO DOCUMENTOS JURÍDICOS")
    logger.info("=" * 50)

    # ✅ CAMINHO CORRETO: data/ está na mesma pasta que scripts/
    base_data_path = "data"

    logger.info(f"📂 Diretório atual: {os.getcwd()}")  # noqa: PTH109
    logger.info(f"🔍 Procurando por: {base_data_path}")

    if not os.path.exists(base_data_path):  # noqa: PTH110
        logger.error(f"❌ Pasta '{base_data_path}' não encontrada!")
        logger.info("📋 Conteúdo do diretório atual:")
        for item in os.listdir("."):  # noqa: PTH208
            if os.path.isdir(item):  # noqa: PTH112
                logger.info(f"   📁 {item}/")
            else:
                logger.info(f"   📄 {item}")
        return

    # ✅ Mapeamento com os nomes exatos das pastas
    document_mapping = {
        "constitution": {
            "domain": "constitutional_law",
            "collection": "constitutional_docs",
            "files": ["CONSTITUICAO.md"],
        },
        "direitos_humanos-oea": {
            "domain": "human_rights_law",
            "collection": "human_rights_docs",
            "files": ["Convencao Americana sobre Direitos Humanos - OEA.pdf"],
        },
        "economia": {
            "domain": "consumer_law",
            "collection": "consumer_docs",
            "files": ["codigo_defesa_consumidor.pdf"],
        },
        # ✅ CORREÇÃO: Isola a Lei de Responsabilidade Fiscal em seu próprio domínio
        # para não "contaminar" a base de conhecimento do agente do consumidor.
        "fiscal": {
            "domain": "fiscal_law",
            "collection": "fiscal_docs",
            "files": ["lei_responsabilidade_fiscal.pdf"],
        },
    }

    processor = DocumentProcessor()
    total_processed = 0

    # Processa cada pasta e arquivo sequencialmente
    for folder, config in document_mapping.items():
        logger.info(f"\n📁 Processando: {folder} -> {config['domain']}")

        folder_path = os.path.join(base_data_path, folder)  # noqa: PTH118

        if not os.path.exists(folder_path):  # noqa: PTH110
            logger.warning(f"   ❌ Pasta não encontrada: {folder_path}")
            continue

        # Processa cada arquivo na pasta
        for filename in config["files"]:
            file_path = os.path.join(folder_path, filename)  # noqa: PTH118

            if os.path.exists(file_path):  # noqa: PTH110
                try:
                    logger.info(f"   📄 Processando: {filename}")

                    result = processor.process_document(
                        file_path=file_path,
                        domain=config["domain"],
                        collection_name=config["collection"],
                        metadata={
                            "source": filename,
                            "domain": config["domain"],
                            "folder": folder,
                        },
                    )

                    if result.get("success"):
                        chunks = result.get("chunks_created", 0)
                        logger.info(f"   ✅ Sucesso: {chunks} chunks criados")
                        total_processed += chunks
                    else:
                        logger.error(f"   ❌ Erro: {result.get('error', 'Desconhecido')}")

                except Exception as e:  # noqa: BLE001
                    logger.error(f"   ❌ Erro inesperado ao processar {filename}: {e}")
            else:
                logger.warning(f"   ❌ Arquivo não encontrado, pulando: {file_path}")

    logger.info("\n🎉 PROCESSAMENTO CONCLUÍDO!")
    logger.info(f"   📊 Total de chunks criados: {total_processed}")

    # Verifica as collections criadas
    chroma_status = check_chroma_connections()

    if chroma_status["status"] == "connected":
        logger.info("   📚 Collections no Chroma:")
        for collection_name, info in chroma_status["collections"].items():
            logger.info(f"      - {collection_name}: {info['count']} documentos")
    else:
        logger.error(f"   ❌ Erro no Chroma: {chroma_status['error']}")


if __name__ == "__main__":
    main()
