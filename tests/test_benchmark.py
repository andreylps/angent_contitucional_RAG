#!/usr/bin/env python3
"""
Sistema de Benchmark Automatizado para RAG Jurídico
Testa performance, inteligência e confiabilidade dos agentes
"""

import os
import sys

# Adiciona o diretório src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))  # noqa: PTH118, PTH120

try:
    from multi_agent_manager_old import MultiAgentManager  # noqa: F401
    from utils.logger import get_logger
except ImportError as e:
    print(f"❌ Erro de import: {e}")
    print("📁 Verifique a estrutura de diretórios e imports")
    sys.exit(1)

logger = get_logger(__name__)

# ... o resto do código permanece igual ...
