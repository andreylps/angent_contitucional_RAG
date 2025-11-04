#!/usr/bin/env python3
"""
Configurador Rápido do Projeto RAG Jurídico
"""

import os


def setup_project():
    print("⚙️ CONFIGURANDO PROJETO RAG JURÍDICO")
    print("=" * 50)

    # 1. Verifica diretório
    current_dir = os.getcwd()
    print(f"📂 Diretório atual: {current_dir}")

    # 2. Cria estrutura de diretórios
    directories = [
        "data/constitution",
        "data/direitos_humanos-oea",
        "data/economia",
        "src/pipelines",
        "src/agents",
        "src/utils",
        "scripts",
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Criado: {directory}")

    # 3. Cria arquivo .env exemplo
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write('OPENAI_API_KEY="sua_chave_aqui"\n')
        print("✅ Arquivo .env criado (configure sua OPENAI_API_KEY)")

    # 4. Verifica dependências
    print("\n🔧 VERIFICANDO DEPENDÊNCIAS:")
    dependencies = ["chromadb", "langchain_core", "langchain_openai", "pypdf"]

    for dep in dependencies:
        try:
            __import__(dep)
            print(f"   ✅ {dep}")
        except ImportError:
            print(f"   ❌ {dep} - Execute: pip install {dep}")

    print("\n🎯 CONFIGURAÇÃO COMPLETA!")
    print("💡 Próximos passos:")
    print("   1. Configure OPENAI_API_KEY no arquivo .env")
    print("   2. Adicione seus documentos PDF/MD nas pastas data/")
    print("   3. Execute: python scripts/process_documents.py")


if __name__ == "__main__":
    setup_project()
