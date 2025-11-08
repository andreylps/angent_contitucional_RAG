from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.multi_agent_manager import MultiAgentManager

# --- Modelos de Dados (Pydantic) ---
# Usamos Pydantic para validação automática de tipos e para a documentação da API.


class QueryRequest(BaseModel):
    """Modelo para a requisição de uma consulta."""

    query: str
    conversation_id: str | None = (
        None  # Opcional, para futuras implementações de multi-usuário
    )


class QueryResponse(BaseModel):
    """Modelo para a resposta da consulta."""

    final_answer: str
    sources: list[str]
    status: str


# --- Inicialização da Aplicação ---

# Cria a aplicação FastAPI
app = FastAPI(
    title="RAG Jurídico API",
    description="API para interagir com o sistema de assistente jurídico multiagente.",
    version="0.6.0",
)

# ✅ Singleton: Instancia o MultiAgentManager uma única vez quando a API inicia.
# Isso evita recarregar todos os modelos a cada requisição, o que é crucial para a performance.
manager = MultiAgentManager()


# --- Endpoints da API ---


@app.post("/query", response_model=QueryResponse)
async def process_user_query(request: QueryRequest):
    """
    Recebe uma pergunta do usuário, processa através do sistema multiagente e retorna a resposta.
    """
    try:
        # Como nosso manager é assíncrono, podemos usar 'await' diretamente.
        result = await manager.process_query(request.query)
        return QueryResponse(
            final_answer=result.get("final_answer", "Erro ao processar a resposta."),
            sources=result.get("sources", []),
            status=result.get("status", "error"),
        )
    except Exception as e:
        # Captura exceções inesperadas e retorna um erro HTTP 500.
        raise HTTPException(status_code=500, detail=f"Erro interno no servidor: {e}")
