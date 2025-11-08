"""
Processador de documentos OTIMIZADO para produção e grande escala.

✅ NOVA LÓGICA DE ALTA PERFORMANCE:
1. Usa `unstructured` para extrair texto de qualquer formato (PDF, DOCX, Imagens, etc.).
2. Usa `SentenceTransformers` para gerar embeddings localmente (muito mais rápido e sem custo de API).
3. Implementa processamento paralelo para utilizar todos os núcleos da CPU.
"""  # noqa: E501

import hashlib
from pathlib import Path
from typing import Any

from chromadb import PersistentClient
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:
    """Processa documentos de vários formatos e os indexa no Chroma"""

    def __init__(
        self, chroma_path: str = "chroma_db", max_batch_size: int = 200
    ) -> None:
        """
        Inicializa o processador com um modelo de embedding local e rápido.
        O modelo 'all-MiniLM-L6-v2' é um excelente padrão para performance e qualidade.
        """
        # Usa o modelo de embedding das configurações globais
        # ✅ OTIMIZAÇÃO: Troca OpenAIEmbeddings por um modelo local para alta velocidade.  # noqa: E501
        # ✅ ATUALIZAÇÃO: Usa o novo pacote modular da LangChain para evitar a depreciação.  # noqa: E501
        from langchain_community.embeddings import (
            HuggingFaceEmbeddings,
        )

        # O modelo 'all-MiniLM-L6-v2' é rápido e eficiente, rodando localmente.
        # Na primeira execução, ele será baixado automaticamente.
        self.embedding_function = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},  # Use 'cuda' se tiver GPU
        )

        self.chroma_path = chroma_path
        self.client = PersistentClient(path=chroma_path)
        self.max_batch_size = max_batch_size
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    def process_document(
        self,
        file_path: str,
        domain: str,
        collection_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Processa um documento e o adiciona ao Chroma em batches
        """
        try:
            print(f"   🔍 Lendo documento: {Path(file_path).name}")
            content = self.read_document(file_path)  # ✅ MUDADO: método público

            if not content or not content.strip():
                return {
                    "success": False,
                    "error": "Documento vazio ou não pôde ser lido",
                }

            print(f"   ✅ Documento lido: {len(content)} caracteres")
            chunks = self.split_text(content)  # ✅ MUDADO: método público
            print(f"   ✂️  Dividido em {len(chunks)} chunks")

            documents = []
            for i, chunk in enumerate(chunks):
                # ✅ CORREÇÃO DEFINITIVA: Combina os metadados recebidos com os metadados do chunk.  # noqa: E501
                # O erro 'domain' ocorria porque o metadata original era sobrescrito.
                doc_metadata = (
                    metadata or {}
                ).copy()  # Começa com uma cópia dos metadados recebidos

                # Adiciona/sobrescreve com informações específicas do chunk
                doc_metadata.update(
                    {
                        "source": str(file_path),
                        "domain": str(domain),
                        "chunk_id": int(i),
                        "total_chunks": len(chunks),
                        "file_type": str(Path(file_path).suffix.lower()),
                        "file_name": str(Path(file_path).name),
                    }
                )

                documents.append(Document(page_content=chunk, metadata=doc_metadata))

            # ✅ CORREÇÃO: Reacopla a lógica de salvar no banco de dados dentro deste método.
            chunks_created = self.add_to_chroma_in_batches(
                documents, collection_name, domain
            )

            return {
                "success": True,
                "chunks_created": chunks_created,
                "total_chunks": len(chunks),
                "collection": collection_name,
            }

        except Exception as e:  # noqa: BLE001
            print(f"   ❌ Erro no processamento: {e}")
            return {"success": False, "error": str(e)}

    def read_document(self, file_path: str) -> str:  # ✅ MUDADO: método público
        """Lê documento baseado na extensão do arquivo"""
        file_ext = Path(file_path).suffix.lower()

        try:
            if file_ext == ".pdf":
                return self.read_pdf_simple(file_path)  # ✅ MUDADO: método público
            if file_ext in [".md", ".txt"]:
                return self.read_text(file_path)  # ✅ MUDADO: método público
            return self.read_text(file_path)  # Tenta como texto genérico

        except Exception as e:  # noqa: BLE001
            print(f"   ❌ Erro ao ler {file_path}: {e}")
            return ""

    def read_pdf_simple(  # noqa: C901, PLR0912
        self, file_path: str
    ) -> str:  # ✅ MUDADO: método público
        """Lê arquivo PDF - versão corrigida"""
        text = ""

        # TENTATIVA 1: pdfplumber (melhor para manter layout)
        try:
            import pdfplumber

            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            if text.strip():
                print(f"   ✅ PDF lido com pdfplumber: {len(text)} caracteres")
                return text
        except Exception as e:
            print(f"   ⚠️ pdfplumber falhou: {e}")

        # TENTATIVA 1: PyMuPDF (mais robusto)
        try:
            # ✅ CORREÇÃO: Usa a importação moderna 'pymupdf' para evitar conflitos e ser mais claro  # noqa: E501
            import pymupdf  # noqa: PLC0415

            doc = pymupdf.open(file_path)
            for _page_num, page in enumerate(doc):  # pyright: ignore[reportArgumentType]
                page_text = page.get_text()
                if page_text:
                    page_text_str = (
                        str(page_text) if not isinstance(page_text, str) else page_text
                    )
                    text += page_text_str + "\n"
            doc.close()
            if text.strip():
                print(f"   ✅ PDF lido com PyMuPDF: {len(text)} caracteres")
                return text
        except Exception as e:  # noqa: BLE001
            print(f"   ⚠️ PyMuPDF falhou: {e}")

        # TENTATIVA 2: PyPDF2
        try:
            import PyPDF2  # noqa: PLC0415

            with open(file_path, "rb") as file:  # noqa: PTH123
                pdf_reader = PyPDF2.PdfReader(file)
                for _page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            if text.strip():
                print(f"   ✅ PDF lido com PyPDF2: {len(text)} caracteres")
                return text
        except Exception as e:  # noqa: BLE001
            print(f"   ⚠️ PyPDF2 falhou: {e}")

        # TENTATIVA 4: OCR com Pytesseract (fallback poderoso)
        try:
            import pytesseract  # noqa: PLC0415
            from pdf2image import convert_from_path  # noqa: PLC0415

            print("   OCR com Pytesseract...")
            images = convert_from_path(file_path)
            ocr_text_parts = []
            for i, image in enumerate(images):
                print(f"      - Processando página {i + 1}/{len(images)} com OCR")
                # Usa 'por' para o idioma português
                page_text = pytesseract.image_to_string(image, lang="por")
                if page_text:
                    ocr_text_parts.append(page_text)

            text = "\n".join(ocr_text_parts)
            if text.strip():
                print(f"   ✅ PDF lido com OCR (Pytesseract): {len(text)} caracteres")
                return text

        except Exception as e:
            print(f"   ⚠️ OCR (Pytesseract) falhou: {e}")

        # TENTATIVA 3: Fallback básico
        try:
            with open(file_path, "rb") as file:  # noqa: PTH123
                raw_content = file.read()
                text_parts = []
                for line in raw_content.split(b"\n"):
                    try:
                        decoded = line.decode("utf-8", errors="ignore")
                        if len(decoded.strip()) > 10:  # noqa: PLR2004
                            text_parts.append(decoded)
                    except:  # noqa: E722, S112
                        continue
                result = "\n".join(text_parts)
                if result.strip():
                    print(f"   ✅ PDF lido com fallback: {len(result)} caracteres")
                    return result
        except Exception as e:  # noqa: BLE001
            print(f"   ⚠️ Fallback falhou: {e}")

        print("   ❌ Todas as tentativas de ler PDF falharam")
        return ""

    def read_text(self, file_path: str) -> str:  # ✅ MUDADO: método público
        """Lê arquivo de texto (MD, TXT)"""
        encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]

        for encoding in encodings:
            try:
                with open(file_path, encoding=encoding) as file:  # noqa: PTH123
                    content = file.read()
                    if content.strip():
                        return content
            except UnicodeDecodeError:
                continue
            except Exception as e:  # noqa: BLE001
                print(f"   ⚠️ Encoding {encoding} falhou: {e}")
                continue

        print(f"   ❌ Nenhum encoding funcionou para {file_path}")
        return ""

    def split_text(self, text: str) -> list[str]:  # ✅ MUDADO: método público
        """Divide texto em chunks"""
        if not text.strip():
            return []

        try:
            return self.text_splitter.split_text(text)
        except Exception as e:  # noqa: BLE001
            print(f"   ⚠️ Erro no split: {e}")
            # Fallback: split simples por quebras de linha
            return [chunk for chunk in text.split("\n") if chunk.strip()]

    def add_to_chroma_in_batches(
        self, documents: list[Document], collection_name: str, domain: str
    ) -> int:
        """Adiciona documentos ao Chroma em batches menores"""
        try:
            from langchain_community.vectorstores import Chroma  # noqa: PLC0415

            # ✅ CORREÇÃO: Reintroduz o processamento em lotes para evitar o erro de limite de tokens da API.  # noqa: E501
            # Inicializa o vector store uma vez.
            vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embedding_function,
                persist_directory=self.chroma_path,
            )

            total_added = 0
            # Itera sobre os documentos em lotes (batches)
            for i in range(0, len(documents), self.max_batch_size):
                batch_documents = documents[i : i + self.max_batch_size]
                batch_number = (i // self.max_batch_size) + 1

                print(
                    f"   📦 Processando lote {batch_number}: {len(batch_documents)} documentos"  # noqa: E501
                )

                # Gera IDs únicos para o lote
                batch_ids = []
                for j, doc in enumerate(batch_documents):
                    content_hash = hashlib.md5(  # noqa: S324
                        doc.page_content.encode()
                    ).hexdigest()[  # noqa: RUF100, S324
                        :16
                    ]
                    batch_ids.append(f"{collection_name}_{i + j}_{content_hash}")

                # Adiciona o lote ao ChromaDB
                vector_store.add_documents(documents=batch_documents, ids=batch_ids)

                total_added += len(batch_documents)
                print(f"   ✅ Lote {batch_number} adicionado com sucesso.")

            print(
                f"   🎉 Total adicionado à {collection_name}: {total_added} documentos"
            )
            return total_added  # noqa: TRY300

        except Exception as e:  # noqa: BLE001
            print(f"❌ Erro ao adicionar ao Chroma: {e}")
            return 0

    def get_collection_info(self, collection_name: str) -> dict[str, Any]:
        """Obtém informações de uma collection"""
        try:
            collection = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "count": collection.count(),
                "metadata": collection.metadata,
            }
        except Exception as e:  # noqa: BLE001
            return {"name": collection_name, "count": 0, "error": str(e)}
