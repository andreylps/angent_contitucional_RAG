#!/usr/bin/env python3
"""
Teste básico para verificar o ambiente
"""

import os

print("🔍 DIAGNÓSTICO DO AMBIENTE")
print("=" * 40)

# 1. Verifica diretório atual
print(f"📂 Diretório atual: {os.getcwd()}")  # noqa: PTH109

# 2. Verifica se data/ existe
data_path = "data"
print(f"📁 Pasta 'data' existe: {os.path.exists(data_path)}")  # noqa: PTH110

if os.path.exists(data_path):  # noqa: PTH110
    print("   Conteúdo de data/:")
    for item in os.listdir(data_path):  # noqa: PTH208
        print(f"   - {item}")

# 3. Tenta importações básicas
print("\n🔧 TESTANDO IMPORTAÇÕES:")
try:
    from src.pipelines.document_processor import DocumentProcessor  # noqa: F401

    print("   ✅ DocumentProcessor importado")
except ImportError as e:
    print(f"   ❌ Erro importando DocumentProcessor: {e}")

# 4. Verifica ChromaDB
try:
    import chromadb  # noqa: F401

    print("   ✅ ChromaDB importado")
except ImportError as e:
    print(f"   ❌ ChromaDB não disponível: {e}")

print("\n🎯 DIAGNÓSTICO COMPLETO")
