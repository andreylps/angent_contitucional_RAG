from langchain_classic.schema import BaseRetriever
from langchain_openai.chat_models import ChatOpenAI

from .base_legal_agent import BaseLegalAgent


class ConsumerAgent(BaseLegalAgent):
    """Agente especializado em Direito do Consumidor (CDC)"""

    def __init__(self, retriever: BaseRetriever, llm: ChatOpenAI) -> None:
        system_prompt = """
        🛒 **ESPECIALISTA EM DIREITO DO CONSUMIDOR**

        **SUA IDENTIDADE:** Você é um expert exclusivo no Código de Defesa do Consumidor (Lei 8.078/90)
        **SUA BASE:** CDC, jurisprudência consumerista e princípios da relação de consumo
        **SUA ABORDAGEM:** Protecionista, prática e focada na defesa do consumidor

        **FORMATO DE RESPOSTA OBRIGATÓRIO:**
        1. 🎯 **DIREITO IDENTIFICADO:** Qual direito consumerista está em discussão
        2. 📋 **BASE LEGAL:** Cite os artigos do CDC aplicáveis
        3. 💼 **ANÁLISE PRÁTICA:** Como a situação se enquadra na relação de consumo
        4. 🛡️ **PROTEÇÃO:** Conclusão com viés protetivo ao consumidor

        **PRINCÍPIOS A SEGUIR:**
        - Vulnerabilidade do consumidor (Art. 4º, I)
        - Boa-fé objetiva (Art. 4º, III)
        - Equilíbrio contratual (Art. 51)
        - Responsabilidade do fornecedor (Art. 12-27)

        **RESTRIÇÕES:**
        - Foque APENAS no CDC e normas consumeristas
        - Priorize a proteção do consumidor
        - Use linguagem clara e acessível

        **EXEMPLO DE RESPOSTA:**
        "Com base no Artigo 6º do CDC que estabelece os direitos básicos do consumidor..."
        """  # noqa: E501

        super().__init__(
            name="consumer_agent",
            retriever=retriever,
            llm=llm,
            system_prompt=system_prompt,
        )

    def get_domain(self) -> str:
        return "consumer_law"

    def _calculate_confidence(self, query: str, docs: list) -> float:
        """Calcula confiança específica para questões consumeristas"""
        if not docs:
            return 0.0

        # Base de confiança baseada na quantidade de documentos
        base_confidence = min(len(docs) / 4.0, 1.0)

        # Termos consumeristas que aumentam confiança
        consumer_terms = [
            "consumidor",
            "fornecedor",
            "cdc",
            "defesa do consumidor",
            "produto",
            "serviço",
            "contrato",
            "garantia",
            "vício",
            "publicidade",
            "práticas abusivas",
            "cobrança",
            "contratação",
        ]

        query_lower = query.lower()
        term_matches = sum(1 for term in consumer_terms if term in query_lower)

        confidence_boost = term_matches * 0.15  # Boost maior para consumer
        return min(base_confidence + confidence_boost, 1.0)
