# src/utils/llm_utils.py

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

# -----------------------------------------------------------------------------
# ⚙️ 1. Carregar variáveis de ambiente
# -----------------------------------------------------------------------------
load_dotenv()


# -----------------------------------------------------------------------------
# 🧠 2. Função utilitária para carregar o modelo da OpenAI
# -----------------------------------------------------------------------------
def load_llm(model_name: str = "gpt-4o-mini", temperature: float = 0.0):  # noqa: ANN201
    """
    Carrega e retorna o modelo da OpenAI configurado.
    Permite fácil alteração de modelo e parâmetros globais.
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        msg = (
            "❌ A variável de ambiente OPENAI_API_KEY não está definida. "
            "Adicione-a no arquivo .env ou exporte no sistema."
        )
        raise ValueError(msg)

    # 🔒 Converter string para SecretStr (para evitar alertas de tipo)
    secret_api_key = SecretStr(api_key)

    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=secret_api_key,
    )

    print(f"🤖 LLM carregado com sucesso: {model_name}")
    return llm
