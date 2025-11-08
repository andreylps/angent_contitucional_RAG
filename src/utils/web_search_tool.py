import os
from typing import Any

from langchain_community.tools.tavily_search import TavilySearchResults


class WebSearchTool:
    """
    Uma ferramenta para realizar buscas na web com foco em fontes confiáveis.
    Utiliza a API da Tavily para buscar e extrair conteúdo de páginas.
    """

    def __init__(self, max_results: int = 3):
        """
        Inicializa a ferramenta de busca.

        Args:
            max_results: O número máximo de resultados a serem retornados.
        """
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            msg = "TAVILY_API_KEY não encontrada no arquivo .env"
            raise ValueError(msg)

        self.search_tool = TavilySearchResults(
            max_results=max_results, api_key=tavily_api_key
        )
        # Lista de sites confiáveis para restringir a busca
        self.trusted_sites = [
            "gov.br",
            "stf.jus.br",
            "stj.jus.br",
            "tse.jus.br",
            "ibge.gov.br",
            "ipea.gov.br",
            "oas.org",  # OEA
            "un.org",  # ONU
        ]

    def search(self, query: str) -> list[dict[str, Any]]:
        """
        Executa uma busca na web, restringindo a sites confiáveis.

        Args:
            query: A pergunta ou termo a ser buscado.

        Returns:
            Uma lista de dicionários, onde cada um contém 'url' e 'content'.
        """
        # Constrói a query de busca com a restrição de sites
        site_query = " OR ".join([f"site:{site}" for site in self.trusted_sites])
        full_query = f"{query} ({site_query})"

        print(f"   [WebSearchTool] Executando busca na web com a query: '{full_query}'")

        # A ferramenta Tavily retorna uma lista de dicionários com 'url' e 'content'
        results = self.search_tool.invoke({"query": full_query})
        return results
