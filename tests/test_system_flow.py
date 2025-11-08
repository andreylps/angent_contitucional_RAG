#!/usr/bin/env python3
"""
Script de teste de fluxo para o sistema RAG Jurídico Adaptativo.

Este script executa uma série de perguntas predefinidas para validar:
1. O roteamento semântico do LegalRouterAgent.
2. A execução de agentes únicos.
3. A execução e síntese de múltiplos agentes.
4. O tratamento de perguntas fora de contexto.
5. O cálculo de confiança e a citação de fontes.
"""

import asyncio
import json
import os
import sys

from rich.console import Console
from rich.panel import Panel

# Adiciona o diretório raiz do projeto ao path para encontrar o módulo 'src'
# O script está em 'tests/', então precisamos subir um nível ('..')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # noqa: PTH100, PTH118, PTH120

from src.multi_agent_manager import MultiAgentManager
from src.utils.logger import logger

console = Console()


async def run_test(manager: MultiAgentManager, question: str):
    """Executa um único teste e imprime o resultado formatado."""
    console.print(
        f"[bold cyan]▶️  Testando pergunta:[/bold cyan] [italic]{question}[/italic]"
    )

    try:
        result = await manager.process_query(question)

        # Formata o resultado para exibição
        routing_decision = result.get("routing_decision", {})
        selected_agents = routing_decision.get("selected_agents", "N/A")

        panel_content = f"""
[bold]Resposta Final:[/bold]
{result.get("final_answer", "N/A")}

---
[bold]Agentes Selecionados:[/bold] {selected_agents}
[bold]Status:[/bold] {result.get("status", "N/A")}
"""

        # Adiciona confiança e fontes se existirem
        if "confidence" in result:
            panel_content += (
                f"\n[bold]Confiança:[/bold] {result.get('confidence', 0.0):.2f}"
            )

        if result.get("sources"):
            sources_str = json.dumps(result["sources"], indent=2, ensure_ascii=False)
            panel_content += f"\n[bold]Fontes:[/bold]\n{sources_str}"

        console.print(Panel(panel_content, title="Resultado do Teste", expand=False))

    except Exception:  # noqa: BLE001
        logger.error(f"❌ Erro catastrófico ao processar a pergunta: {question}")
        console.print_exception()


async def main():
    """Função principal que executa todos os testes."""
    console.rule(
        "[bold green]🚀 Iniciando Bateria de Testes do Sistema RAG 🚀[/bold green]"
    )
    manager = MultiAgentManager()

    test_questions = [
        # 1. Teste de agente único (Constitucional)
        "O que é o princípio da dignidade da pessoa humana na CF/88?",
        # 2. Teste de agente único (Consumidor)
        "Quais são os meus direitos se um produto que comprei veio com defeito?",
        # 3. Teste de múltiplos agentes (Constitucional + Consumidor)
        "Como a Constituição Federal protege os direitos do consumidor?",
        # 4. Teste de pergunta fora de contexto
        "Qual a receita de bolo de fubá?",
        # 5. Teste que pode gerar baixa confiança
        "Qual a jurisprudência do STF sobre criptomoedas em 2025?",
    ]

    for question in test_questions:
        await run_test(manager, question)


if __name__ == "__main__":
    asyncio.run(main())
