#!/usr/bin/env python3
"""
Script para processar documentos jurídicos e criar collections no Chroma
"""

import os
import sys

# Adiciona o src ao path para imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # noqa: PTH118, PTH120

from src.pipelines.document_processor import DocumentProcessor


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """Processa todos os documentos jurídicos"""
    print("📚 PROCESSANDO DOCUMENTOS JURÍDICOS")
    print("=" * 50)

    # ✅ CAMINHO CORRETO: data/ está na mesma pasta que scripts/
    base_data_path = "data"

    print(f"📂 Diretório atual: {os.getcwd()}")  # noqa: PTH109
    print(f"🔍 Procurando por: {base_data_path}")

    if not os.path.exists(base_data_path):  # noqa: PTH110
        print(f"❌ Pasta '{base_data_path}' não encontrada!")
        print("📋 Conteúdo do diretório atual:")
        for item in os.listdir("."):  # noqa: PTH208
            if os.path.isdir(item):  # noqa: PTH112
                print(f"   📁 {item}/")
            else:
                print(f"   📄 {item}")
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
            "files": [
                "codigo_defesa_consumidor.pdf",
                "lei_responsabilidade_fiscal.pdf",
            ],
        },
    }

    processor = DocumentProcessor()
    total_processed = 0

    for folder, config in document_mapping.items():
        print(f"\n📁 Processando: {folder} -> {config['domain']}")

        folder_path = os.path.join(base_data_path, folder)  # noqa: PTH118

        if not os.path.exists(folder_path):  # noqa: PTH110
            print(f"   ❌ Pasta não encontrada: {folder_path}")
            print(f"   🔍 Pastas disponíveis em '{base_data_path}/':")
            try:
                available_folders = os.listdir(base_data_path)  # noqa: PTH208
                for available_folder in available_folders:
                    print(f"      - {available_folder}")
            except Exception as e:  # noqa: BLE001
                print(f"      Erro: {e}")
            continue

        # Processa cada arquivo na pasta
        for filename in config["files"]:
            file_path = os.path.join(folder_path, filename)  # noqa: PTH118

            if not os.path.exists(file_path):  # noqa: PTH110
                print(f"   ❌ Arquivo não encontrado: {filename}")
                print(f"   🔍 Arquivos disponíveis em '{folder}/':")
                try:
                    available_files = os.listdir(folder_path)  # noqa: PTH208
                    for available_file in available_files:
                        print(f"      - {available_file}")
                except Exception as e:  # noqa: BLE001
                    print(f"      Erro: {e}")
                continue

            try:
                print(f"   📄 Processando: {filename}")

                # Processa o documento
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

                if result["success"]:
                    print(f"   ✅ Sucesso: {result['chunks_created']} chunks criados")
                    total_processed += result["chunks_created"]
                else:
                    print(f"   ❌ Erro: {result['error']}")

            except Exception as e:  # noqa: BLE001
                print(f"   ❌ Erro ao processar {filename}: {e}")

    print("\n🎉 PROCESSAMENTO CONCLUÍDO!")
    print(f"   📊 Total de chunks criados: {total_processed}")

    # Verifica as collections criadas
    from src.pipelines.specialized_retrievers import (  # noqa: PLC0415
        check_chroma_connections,
    )

    chroma_status = check_chroma_connections()

    if chroma_status["status"] == "connected":
        print("   📚 Collections no Chroma:")
        for collection_name, info in chroma_status["collections"].items():
            print(f"      - {collection_name}: {info['count']} documentos")
    else:
        print(f"   ❌ Erro no Chroma: {chroma_status['error']}")


if __name__ == "__main__":
    main()
