#!/usr/bin/env python3
"""
Teste diagnóstico completo
"""

import os
import sys


def diagnose():
    print("🔍 DIAGNÓSTICO COMPLETO DO SISTEMA")
    print("=" * 50)

    # 1. Diretórios
    current_dir = os.getcwd()  # noqa: PTH109
    print(f"📂 Diretório atual: {current_dir}")

    # 2. Estrutura de arquivos
    print("\n📁 ESTRUTURA DO PROJETO:")
    for item in os.listdir(current_dir):  # noqa: PTH208
        item_path = os.path.join(current_dir, item)  # noqa: PTH118
        if os.path.isdir(item_path):  # noqa: PTH112
            print(f"   📁 {item}/")
            # Mostra conteúdo de pastas importantes
            if item in ["src", "scripts", "data"]:
                try:
                    subitems = os.listdir(item_path)  # noqa: PTH208
                    for subitem in subitems:
                        print(f"      📄 {subitem}")
                except Exception as e:  # noqa: BLE001
                    print(f"      ❌ Erro: {e}")
        else:
            print(f"   📄 {item}")

    # 3. Teste de importações
    print("\n🔧 TESTE DE IMPORTAÇÕES:")

    # Adiciona src ao path
    src_path = os.path.join(current_dir, "src")  # noqa: PTH118
    sys.path.insert(0, src_path)
    print(f"   Python path configurado: {src_path}")

    # Testa importação
    try:
        from pipelines.document_processor import (  # noqa: PLC0415
            DocumentProcessor,  # noqa: F401
        )

        print("   ✅ from pipelines.document_processor import DocumentProcessor")
    except ImportError as e:
        print(f"   ❌ Erro: {e}")

    try:
        from pipelines import document_processor  # noqa: F401, PLC0415

        print("   ✅ from pipelines import document_processor")
    except ImportError as e:
        print(f"   ❌ Erro: {e}")

    # 4. Testa se arquivos existem
    print("\n📄 VERIFICAÇÃO DE ARQUIVOS:")
    important_files = [
        "src/pipelines/document_processor.py",
        "src/pipelines/specialized_retrievers.py",
        "scripts/process_documents.py",
        ".env",
    ]

    for file_path in important_files:
        exists = os.path.exists(file_path)  # noqa: PTH110
        status = "✅" if exists else "❌"
        print(f"   {status} {file_path}")


if __name__ == "__main__":
    diagnose()
