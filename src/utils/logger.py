#!/usr/bin/env python3
"""
Sistema de logging para o projeto RAG Jurídico
"""

import logging
from pathlib import Path

# Define o caminho para logs
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "app.log"

# Configuração básica do logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)

# Cria instância global de logger
logger = logging.getLogger("constitutional_rag")

# Exemplo opcional de mensagem inicial
logger.info("✅ Logger inicializado com sucesso.")
