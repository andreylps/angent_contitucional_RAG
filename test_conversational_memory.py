import asyncio
from pprint import pprint

from src.multi_agent_manager import MultiAgentManager


async def run_test():
    """
    Executa um teste de conversação para validar a memória (v0.3).
    """
    print("🚀 Iniciando teste da v0.3 - Memória Conversacional...")
    print("=" * 60)

    # 1. Inicializa o gerenciador de agentes.
    # A mesma instância será usada para toda a conversa.
    manager = MultiAgentManager()

    # 2. Primeira pergunta (estabelece o contexto)
    query1 = "Quais são os direitos fundamentais na Constituição Federal?"
    print("\n--- TURNO 1: Pergunta Inicial ---")
    result1 = await manager.process_query(query1)
    print("\n✅ Resultado do Turno 1:")
    pprint(result1["final_answer"])

    # 3. Segunda pergunta (pergunta de acompanhamento que depende do histórico)
    query2 = "E quais são as principais garantias para esses direitos?"
    print("\n\n--- TURNO 2: Pergunta de Acompanhamento ---")
    result2 = await manager.process_query(query2)
    print("\n✅ Resultado do Turno 2:")
    pprint(result2["final_answer"])

    # Opcional: Limpar o histórico para iniciar uma nova conversa
    manager.clear_history()

    print("\n" + "=" * 60)
    print("✅ Teste de conversação concluído.")


if __name__ == "__main__":
    asyncio.run(run_test())
