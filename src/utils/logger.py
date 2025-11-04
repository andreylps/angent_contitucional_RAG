#!/usr/bin/env python3
"""
Sistema de logging para o projeto RAG Jurídico
"""

import logging
import os
import sys


def setup_logger(
    name: str = "rag_juridico", level: int = logging.INFO
) -> logging.Logger:
    """
    Configura e retorna um logger

    Args:
        name: Nome do logger
        level: Nível de log

    Returns:
        Logger configurado
    """
    logger = logging.getLogger(name)

    # Evita múltiplos handlers
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Formatação
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler para arquivo (opcional)
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)  # noqa: PTH103

    file_handler = logging.FileHandler(os.path.join(log_dir, "rag_system.log"))  # noqa: PTH118
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# Logger global
logger = setup_logger()
