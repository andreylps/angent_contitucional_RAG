#!/usr/bin/env python3
"""
Sistema RAG Jurídico Multiagente - Interface de Uso CORRIGIDA
"""

import asyncio
import os
import sys
from typing import Any

# Adiciona o src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # noqa: PTH118, PTH120

from src.multi_agent_manager import MultiAgentManager
from src.utils.logger import logger  # ✅ NOVO: Sistema de logging


class RAGJuridicoSystem:
    """Sistema RAG jurídico com interface amigável"""

    def __init__(self) -> None:
        self.manager: MultiAgentManager | None = None
        self.is_initialized = False

    async def initialize(self) -> bool:
        """Inicializa o sistema multiagente"""
        print("⚖️  INICIALIZANDO SISTEMA JURÍDICO MULTIAGENTE")
        print("=" * 50)

        try:
            logger.debug("Carregando agentes especializados...")
            self.manager = MultiAgentManager()

            if self.manager is None:
                logger.error("Falha na criação do MultiAgentManager")
                return False

            # Verifica se os agentes foram carregados
            agent_info = self.manager.get_agent_info()

            # ✅ CORREÇÃO: Logs controlados por DEBUG_MODE
            logger.info("Sistema inicializado com sucesso!")
            logger.debug(f"Agentes carregados: {len(agent_info['available_agents'])}")
            logger.debug(f"Modelo: {agent_info['llm_model']}")

            self.is_initialized = True
            return True  # noqa: TRY300

        except Exception as e:  # noqa: BLE001
            logger.error(f"Erro na inicialização: {e}")
            return False

    async def process_question(self, question: str) -> dict[str, Any]:
        """Processa uma pergunta jurídica"""
        if not self.is_initialized or self.manager is None:
            return {"error": "Sistema não inicializado corretamente"}

        try:
            # ✅ CORREÇÃO: NÃO imprime a pergunta aqui (evita duplicação)
            logger.debug(f"Processando pergunta: {question}")
            return await self.manager.process_query(question)

        except Exception as e:  # noqa: BLE001
            logger.error(f"Erro no processamento: {e!s}")
            return {"error": f"Erro no processamento: {e!s}"}

    def get_agent_info(self) -> dict[str, Any]:
        """Obtém informações dos agentes"""
        if not self.is_initialized or self.manager is None:
            return {"error": "Sistema não inicializado"}

        try:
            return self.manager.get_agent_info()
        except Exception as e:  # noqa: BLE001
            logger.error(f"Erro ao obter informações: {e!s}")
            return {"error": f"Erro ao obter informações: {e!s}"}

    def format_response(self, response: dict[str, Any]) -> str:
        """Formata a resposta para exibição"""
        if "error" in response:
            return f"❌ {response['error']}"

        formatted = []

        # ✅ CORREÇÃO: Resposta limpa sem duplicação
        formatted.append("📝 **RESPOSTA:**")
        formatted.append(f"{response['final_answer']}")

        # ✅ CORREÇÃO: Mostra detalhes apenas se não for "fora do escopo"
        if response.get("status") != "out_of_scope" and response.get(
            "routing_decision"
        ):
            formatted.append("")  # Linha em branco
            formatted.append("🎯 **DETALHES:**")
            formatted.append(
                f"   • Agente principal: {response.get('primary_agent', 'N/A')}"
            )

            # Domínios envolvidos
            domains = response.get(
                "agent_domains", [response.get("agent_domain", "N/A")]
            )
            if isinstance(domains, str):
                domains = [domains]
            formatted.append(f"   • Domínios: {', '.join(domains)}")
            formatted.append(f"   • Confiança: {response.get('confidence', 0):.2f}")

            # ✅ CORREÇÃO: Trata a lista de fontes corretamente
            sources = response.get("sources", [])
            if sources:
                # Remove duplicatas baseadas no nome do arquivo
                unique_source_files = sorted(
                    {s.get("file_name") for s in sources if s.get("file_name")}
                )
                formatted.append(
                    f"   • Fontes consultadas: {len(unique_source_files)} ({', '.join(unique_source_files)})"  # noqa: E501
                )
            else:
                formatted.append("   • Fontes consultadas: 0")

            # Decisão de roteamento (se disponível)
            if response.get("routing_decision"):
                rd = response["routing_decision"]
                formatted.append(f"   • Roteamento: {rd.get('selected_agents', [])}")

        return "\n".join(formatted)


async def main() -> None:  # noqa: C901, PLR0915
    """Função principal do sistema - CORRIGIDA"""
    system = RAGJuridicoSystem()

    # Inicializa o sistema
    success = await system.initialize()
    if not success:
        print("❌ Não foi possível inicializar o sistema.")
        print("💡 Verifique se:")
        print("   - O ChromaDB está com as collections")
        print("   - A API key da OpenAI está no .env")
        print("   - As dependências estão instaladas")
        return

    print("\n" + "=" * 50)
    print("✅ SISTEMA PRONTO PARA USO!")
    print("=" * 50)
    print("\n💡 **COMANDOS DISPONÍVEIS:**")
    print("   • Faça perguntas jurídicas")
    print("   • 'agentes' - Lista agentes disponíveis")
    print("   • 'exemplo' - Mostra exemplos de uso")
    print("   • 'sair' - Encerra o sistema")
    print("\n🎯 **EXEMPLOS DE PERGUNTAS:**")
    print("   • 'Quais são os direitos fundamentais?'")
    print("   • 'O que diz o CDC sobre garantia?'")
    print("   • 'Como a Convenção Americana protege direitos humanos?'")
    print("   • 'Qual a diferença entre direitos constitucionais e humanos?'")
    print("-" * 50)

    while True:
        try:
            # ✅ CORREÇÃO PRINCIPAL: Input limpo
            user_input = input("\n🔍 Sua pergunta: ").strip()

            if not user_input:
                continue

            # Comandos especiais
            if user_input.lower() in ["sair", "exit", "quit"]:
                print("👋 Encerrando sistema...")
                break

            if user_input.lower() == "agentes":
                agent_info = system.get_agent_info()
                if "error" in agent_info:
                    print(f"❌ {agent_info['error']}")
                else:
                    print("\n🤖 **AGENTES ESPECIALIZADOS:**")
                    for agent in agent_info.get("available_agents", []):
                        print(f"   • {agent}")
                continue

            if user_input.lower() == "exemplo":
                print("\n🎯 **EXEMPLOS DE CONSULTAS:**")
                examples = [
                    "Quais são os direitos fundamentais na Constituição?",
                    "O que é o direito de arrependimento no CDC?",
                    "Como a Convenção Americana protege a liberdade de expressão?",
                    "Qual a proteção constitucional ao consumidor?",
                    "Direitos humanos na CADH vs direitos fundamentais na CF",
                ]
                for i, example in enumerate(examples, 1):
                    print(f"   {i}. {example}")
                continue

            # ✅ CORREÇÃO: Processa e mostra APENAS a resposta formatada
            response = await system.process_question(user_input)
            formatted_response = system.format_response(response)

            print(f"\n{formatted_response}")

        except KeyboardInterrupt:
            print("\n\n👋 Encerrado pelo usuário.")
            break
        except Exception as e:  # noqa: BLE001
            logger.error(f"Erro: {e}")


if __name__ == "__main__":
    # Verifica se o sistema está pronto
    print("🔍 Verificando se o sistema está pronto...")

    # Verifica se o Chroma tem collections
    try:
        from src.pipelines.specialized_retrievers import check_chroma_connections

        chroma_status = check_chroma_connections()

        if chroma_status["status"] == "connected" and chroma_status["collections"]:
            logger.debug("ChromaDB encontrado com collections:")
            for collection_name, info in chroma_status["collections"].items():
                logger.debug(f"   📚 {collection_name}: {info['count']} documentos")

            # Executa o sistema
            asyncio.run(main())
        else:
            print("❌ ChromaDB não está pronto.")
            print("💡 Execute primeiro: python scripts/process_documents.py")

    except Exception as e:  # noqa: BLE001
        logger.error(f"Erro ao verificar sistema: {e}")
        print("💡 Execute primeiro: python scripts/process_documents.py")
