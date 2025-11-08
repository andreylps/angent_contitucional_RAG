import asyncio
from pprint import pprint

from src.multi_agent_manager import MultiAgentManager


async def run_test():
    """
    Executa um teste focado no ConsumerAgent com a nova lógica Multi-Query.
    """
    print("🚀 Iniciando teste da v0.1 - ConsumerAgent com Multi-Query...")
    print("=" * 60)

    # 1. Inicializa o gerenciador de agentes
    manager = MultiAgentManager()

    # 2. Define uma pergunta que será roteada para o ConsumerAgent
    query = "Quais são os meus direitos se um produto que comprei veio com defeito?"

    # 3. Processa a consulta
    result = await manager.process_query(query)

    print("\n" + "=" * 60)
    print("✅ Teste concluído. Resultado final:")
    pprint(result)


if __name__ == "__main__":
    asyncio.run(run_test())
