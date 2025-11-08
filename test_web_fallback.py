import asyncio
from pprint import pprint

from src.multi_agent_manager import MultiAgentManager


async def run_test():
    """
    Executa um teste para validar o fallback de busca na web (v0.5).
    """
    print("🚀 Iniciando teste da v0.5 - Fallback com Busca na Web...")
    print("=" * 60)

    # 1. Inicializa o gerenciador de agentes.
    manager = MultiAgentManager()

    # 2. Pergunta sobre um tópico que NÃO está na base de dados interna.
    # Isso forçará o agente a usar a WebSearchTool.
    query = "O que diz o Marco Legal das Criptomoedas (Lei 14.478/2022) no Brasil?"
    print(f"🔍 Processando consulta: '{query}'")

    # 3. Processa a consulta
    result = await manager.process_query(query)

    print("\n" + "=" * 60)
    print("✅ Teste de fallback na web concluído. Resultado final:")
    pprint(result)


if __name__ == "__main__":
    asyncio.run(run_test())
