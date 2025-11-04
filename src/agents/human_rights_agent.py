from langchain_classic.schema import BaseRetriever
from langchain_openai.chat_models import ChatOpenAI

from .base_legal_agent import BaseLegalAgent


class HumanRightsAgent(BaseLegalAgent):
    """Agente especializado em Direitos Humanos (Convenção Americana - CADH)"""

    def __init__(self, retriever: BaseRetriever, llm: ChatOpenAI) -> None:
        system_prompt = """
        🌎 **ESPECIALISTA EM DIREITOS HUMANOS - CONVENÇÃO AMERICANA**

        **SUA IDENTIDADE:** Você é um expert exclusivo na Convenção Americana de Direitos Humanos (Pacto de San José da Costa Rica)
        **SUA BASE:** CADH, jurisprudência da Corte Interamericana e princípios internacionais de direitos humanos
        **SUA ABORDAGEM:** Universalista, protetiva e alinhada com os tratados internacionais

        **FORMATO DE RESPOSTA OBRIGATÓRIO:**
        1. 🌐 **DIREITO INTERNACIONAL:** Identifique o direito humano em discussão
        2. 📜 **BASE CONVENCIONAL:** Cite os artigos da CADH aplicáveis
        3. ⚖️ **INTERPRETAÇÃO:** Contextualize com jurisprudência interamericana
        4. 🕊️ **PROTEÇÃO INTEGRAL:** Conclusão com perspectiva universalista

        **PRINCÍPIOS FUNDAMENTAIS:**
        - Dignidade da pessoa humana
        - Não-discriminação
        - Efetividade dos direitos
        - Interpretação pro persona
        - Controle de convencionalidade

        **DIREITOS PROTEGIDOS (EXEMPLOS):**
        - Vida, integridade pessoal (Art. 4º, 5º)
        - Liberdade pessoal (Art. 7º)
        - Garantias judiciais (Art. 8º)
        - Liberdade de consciência e religião (Art. 12º)
        - Liberdade de associação (Art. 16º)
        - Proteção da família (Art. 17º)
        - Direito à propriedade (Art. 21º)

        **RESTRIÇÕES:**
        - Foque na CADH e sistema interamericano
        - Considere a jurisprudência da Corte IDH
        - Use perspectiva internacionalista

        **EXEMPLO DE RESPOSTA:**
        "Com base no Artigo 8º da CADH que garante as garantias judiciais..."
        """  # noqa: E501

        super().__init__(
            name="human_rights_agent",
            retriever=retriever,
            llm=llm,
            system_prompt=system_prompt,
        )

    def get_domain(self) -> str:
        return "human_rights_law"

    def _calculate_confidence(self, query: str, docs: list) -> float:
        """Calcula confiança específica para questões de direitos humanos"""
        if not docs:
            return 0.0

        # Base de confiança baseada na quantidade de documentos
        base_confidence = min(len(docs) / 4.0, 1.0)

        # Termos de direitos humanos que aumentam confiança
        human_rights_terms = [
            "direitos humanos",
            "convenção americana",
            "cadh",
            "corte interamericana",
            "pacto de san josé",
            "direito internacional",
            "tratado internacional",
            "dignidade",
            "liberdade",
            "igualdade",
            "discriminação",
            "vida",
            "integridade",
            "jurisprudência interamericana",
            "sistema interamericano",
        ]

        query_lower = query.lower()
        term_matches = sum(1 for term in human_rights_terms if term in query_lower)

        confidence_boost = term_matches * 0.12
        return min(base_confidence + confidence_boost, 1.0)
