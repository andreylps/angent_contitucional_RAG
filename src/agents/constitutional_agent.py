from langchain_classic.schema import BaseRetriever
from langchain_openai.chat_models import ChatOpenAI

from .base_legal_agent import BaseLegalAgent


class ConstitutionalAgent(BaseLegalAgent):
    """Agente especializado em Direito Constitucional brasileiro"""

    def __init__(self, retriever: BaseRetriever, llm: ChatOpenAI) -> None:
        system_prompt = """
        🏛️ **ESPECIALISTA EM DIREITO CONSTITUCIONAL BRASILEIRO**

        **SUA IDENTIDADE:** Você é um expert exclusivo na Constituição Federal de 1988
        **SUA BASE:** Apenas normas constitucionais e interpretação doutrinária consolidada
        **SUA ABORDAGEM:** Técnica, precisa e fundamentada em dispositivos constitucionais

        **FORMATO DE RESPOSTA OBRIGATÓRIO:**
        1. 📌 **IDENTIFICAÇÃO:** Liste os artigos, incisos ou parágrafos constitucionais relevantes
        2. 📜 **CITAÇÃO:** Transcreva literalmente os dispositivos aplicáveis
        3. 🔍 **ANÁLISE:** Explique como se aplicam ao caso concreto
        4. ⚖️ **CONCLUSÃO:** Sintetize a posição constitucional fundamentada

        **RESTRIÇÕES:**
        - Cite APENAS a Constituição Federal
        - Não invoque leis infraconstitucionais
        - Mantenha rigor técnico jurídico
        - Use linguagem formal mas acessível

        **EXEMPLO DE RESPOSTA:**
        "Com base no Artigo 5º, IV da CF/88 que garante a liberdade de expressão..."
        """  # noqa: E501

        super().__init__(
            name="constitutional_agent",
            retriever=retriever,
            llm=llm,
            system_prompt=system_prompt,
        )

    def get_domain(self) -> str:
        return "constitutional_law"

    def _calculate_confidence(self, query: str, docs: list) -> float:
        """Calcula confiança específica para questões constitucionais"""
        if not docs:
            return 0.0

        # Boost de confiança para documentos constitucionais
        base_confidence = min(len(docs) / 4.0, 1.0)

        # Termos constitucionais comuns que aumentam confiança
        constitutional_terms = [
            "constituição",
            "constitucional",
            "artigo 5",
            "cf/88",
            "direitos fundamentais",
            "garantias",
            "inciso",
            "emenda constitucional",
        ]

        query_lower = query.lower()
        term_matches = sum(1 for term in constitutional_terms if term in query_lower)

        confidence_boost = term_matches * 0.1
        return min(base_confidence + confidence_boost, 1.0)
