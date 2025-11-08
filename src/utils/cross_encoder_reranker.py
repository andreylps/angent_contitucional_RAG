from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


class CrossEncoderReRanker:
    """
    Uma classe para reordenar documentos usando um modelo Cross-Encoder.

    Esta abordagem é mais precisa que o re-ranking baseado em LLM e mais
    eficiente em termos de custo e velocidade para a tarefa de relevância.
    """

    def __init__(
        self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", top_n: int = 4
    ):
        """
        Inicializa o re-ranker com um modelo Cross-Encoder.

        Args:
            model_name: O nome do modelo a ser carregado do HuggingFace.
            top_n: O número de documentos a serem retornados após o re-ranking.
        """
        self.model = CrossEncoder(model_name)
        self.top_n = top_n

    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        """
        Reordena uma lista de documentos com base em sua relevância para uma consulta.
        """
        if not documents:
            return []

        # Cria pares de (query, document_text) para o modelo pontuar
        pairs = [(query, doc.page_content) for doc in documents]

        # Calcula os scores de relevância
        scores = self.model.predict(pairs)

        # Combina documentos e scores, e ordena do maior para o menor score
        doc_scores = list(zip(documents, scores, strict=False))
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        # Retorna os 'top_n' documentos mais relevantes
        reranked_docs = [doc for doc, score in doc_scores[: self.top_n]]
        return reranked_docs
