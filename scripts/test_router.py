# scripts/test_router.py
import asyncio
import os
import sys

from langchain_openai import ChatOpenAI  # pyright: ignore[reportMissingImports]

# Adiciona o diretório raiz do projeto ao path para encontrar o módulo 'src'
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # noqa: PTH118, PTH120
from src.agents.router_agent import OUT_OF_CONTEXT_MESSAGE, LegalRouterAgent
from src.config.setting import Settings  # pyright: ignore[reportMissingImports]
from src.utils.logger import logger


async def main_test() -> None:
    """
    Função principal para testar o agente roteador de forma interativa.
    """
    logger.info("🚀 Iniciando o modo de teste interativo do Roteador Jurídico...")

    # Carrega as configurações
    settings = Settings()
    if not settings.validate():
        logger.error("❌ Configurações inválidas. Abortando o teste.")
        return

    # Inicializa o LLM para o roteador
    router_llm = ChatOpenAI(
        model=settings.llm.router_model,
        temperature=0,
        api_key=settings.llm.api_key,
    )

    # Cria a instância do agente roteador
    router_agent = LegalRouterAgent(llm=router_llm)

    print("\n✅ Sistema de teste pronto. Digite sua pergunta ou 'sair' para terminar.")
    print("-" * 70)

    while True:
        try:
            # Obtém a pergunta do usuário
            query = input(" Pergunte-me algo sobre Direito: ")
            if query.lower() in ["sair", "exit", "quit"]:
                break
            if not query:
                continue

            # Roda a lógica de roteamento
            routing_decision = router_agent.get_routing_decision(query)

            # Exibe o resultado
            print("-" * 70)
            logger.info(f"🎯 Decisão do Roteador: {routing_decision}")

            # Verifica se a resposta foi a de "fora de contexto"
            if routing_decision.get("selected_agents") == ["out_of_context"]:
                print(f"\n🤖 Resposta do Sistema:\n   {OUT_OF_CONTEXT_MESSAGE}\n")
            else:
                # Em um cenário real, aqui você chamaria os agentes selecionados.
                # Para este teste, vamos apenas mostrar a decisão.
                print(
                    f"\n🤖 O roteador direcionou a pergunta para o(s) agente(s): "
                    f"{routing_decision.get('selected_agents')}"
                )
                print("   (Em um sistema completo, a resposta seria gerada agora.)\n")

            print("-" * 70)

        except Exception as e:  # noqa: BLE001
            logger.error(f"💥 Ocorreu um erro durante o teste: {e}")
            break

    logger.info("👋 Encerrando o sistema de teste.")


if __name__ == "__main__":
    asyncio.run(main_test())
