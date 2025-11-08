import asyncio
from pprint import pprint

from src.multi_agent_manager import MultiAgentManager


async def run_test():
    """
    Executa um teste focado no HumanRightsAgent com a nova lógica Multi-Query.
    """
    print("🚀 Iniciando teste da v0.1 - HumanRightsAgent com Multi-Query...")
    print("=" * 60)

    # 1. Inicializa o gerenciador de agentes
    manager = MultiAgentManager()

    # 2. Define uma pergunta que será roteada para o HumanRightsAgent
    query = "Como a Convenção Americana de Direitos Humanos protege a liberdade de expressão?"

    # 3. Processa a consulta
    result = await manager.process_query(query)

    print("\n" + "=" * 60)
    print("✅ Teste concluído. Resultado final:")
    pprint(result)


if __name__ == "__main__":
    asyncio.run(run_test())
