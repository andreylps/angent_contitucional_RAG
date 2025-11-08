import asyncio

from src.multi_agent_manager import MultiAgentManager


async def run_all_tests() -> None:
    """
    Executa um teste para cada agente especializado para validar a arquitetura v0.3.
    """
    print("🚀 Iniciando suíte de testes para todos os agentes (v0.3)...")
    print("=" * 70)

    # 1. Inicializa o gerenciador de agentes.
    manager = MultiAgentManager()

    # --- Teste 1: ConstitutionalAgent ---
    print("\n--- 🏛️  TESTE 1: CONSTITUTIONAL AGENT ---")
    query1 = "Quais são os direitos fundamentais e garantias na Constituição Federal?"
    await manager.process_query(query1)
    print("\n✅ Resultado do Teste Constitucional:")
    manager.clear_history()  # Limpa o histórico para o próximo teste

    # --- Teste 2: ConsumerAgent ---
    print("\n\n--- 🛒 TESTE 2: CONSUMER AGENT ---")
    query2 = "Quais são os meus direitos se um produto que comprei veio com defeito?"
    await manager.process_query(query2)
    print("\n✅ Resultado do Teste do Consumidor:")
    manager.clear_history()  # Limpa o histórico para o próximo teste

    # --- Teste 3: HumanRightsAgent ---
    print("\n\n--- 🕊️  TESTE 3: HUMAN RIGHTS AGENT ---")
    query3 = "Como a Convenção Americana de Direitos Humanos protege a liberdade de expressão?"  # noqa: E501
    await manager.process_query(query3)
    print("\n✅ Resultado do Teste de Direitos Humanos:")
    manager.clear_history()

    print("\n" + "=" * 70)
    print("✅ Suíte de testes concluída com sucesso!")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
