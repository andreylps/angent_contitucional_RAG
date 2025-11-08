import json
from pathlib import Path
from typing import Any

# Define o caminho da raiz do projeto de forma robusta
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
INTERACTION_LOG_FILE = LOG_DIR / "interactions.jsonl"

# ✅ v0.4 CORREÇÃO FINAL: Simplifica o logger para evitar conflitos de configuração.
# Em vez de criar um novo logger, usamos uma função direta para escrever no arquivo.
# Isso é mais robusto e evita problemas com a configuração global do logging.


def log_interaction(data: dict[str, Any]) -> None:
    """Registra um dicionário de dados como uma linha JSON no arquivo de log."""
    try:
        json_record = json.dumps(data, ensure_ascii=False)
        # Escreve diretamente no arquivo, garantindo que a operação seja atômica.
        with INTERACTION_LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json_record + "\n")
    except (OSError, TypeError) as e:
        print(f"Erro ao serializar log de interação para JSON: {e}")
