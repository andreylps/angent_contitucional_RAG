"""
Processador de documentos com suporte a batches grandes
"""

import hashlib
from pathlib import Path
from typing import Any

from chromadb import PersistentClient
from langchain_classic.schema import Document
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter


class DocumentProcessor:
    """Processa documentos de vários formatos e os indexa no Chroma"""

    def __init__(
        self, chroma_path: str = "chroma_db", max_batch_size: int = 5000
    ) -> None:
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
                doc_metadata = {
                    "source": str(file_path),
                    "domain": str(domain),
                    "chunk_id": int(i),
                    "total_chunks": len(chunks),
                    "file_type": str(Path(file_path).suffix.lower()),
                    "file_name": str(Path(file_path).name),
                }

                if metadata:
                    doc_metadata.update(metadata)

                documents.append(Document(page_content=chunk, metadata=doc_metadata))

            # Adiciona em batches menores
            chunks_created = self._add_to_chroma_in_batches(
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

        # TENTATIVA 1: PyMuPDF (mais robusto)
        try:
            import fitz  # noqa: PLC0415

            doc = fitz.open(file_path)
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

    def _add_to_chroma_in_batches(
        self, documents: list[Document], collection_name: str, domain: str
    ) -> int:
        """Adiciona documentos ao Chroma em batches menores"""
        try:
            collections = self.client.list_collections()
            collection_names = [col.name for col in collections]

            if collection_name in collection_names:
                collection = self.client.get_collection(collection_name)
                print(f"   📚 Usando collection existente: {collection_name}")
            else:
                collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"domain": domain, "type": "legal_documents"},
                )
                print(f"   📚 Nova collection criada: {collection_name}")

            total_added = 0

            # Divide os documentos em batches menores
            for batch_start in range(0, len(documents), self.max_batch_size):
                batch_end = min(batch_start + self.max_batch_size, len(documents))
                batch_documents = documents[batch_start:batch_end]

                print(
                    f"   📦 Processando batch {batch_start // self.max_batch_size + 1}: {len(batch_documents)} documentos"  # noqa: E501
                )

                # Prepara dados para Chroma
                documents_texts = [doc.page_content for doc in batch_documents]
                documents_metadatas = []
                documents_ids = []

                for i, doc in enumerate(batch_documents):
                    # Garante tipos primitivos nos metadados
                    clean_metadata = {}
                    for key, value in doc.metadata.items():
                        if isinstance(value, (str, int, float, bool)):
                            clean_metadata[key] = value
                        else:
                            clean_metadata[key] = str(value)

                    documents_metadatas.append(clean_metadata)
                    content_hash = hashlib.md5(  # noqa: S324
                        doc.page_content.encode()
                    ).hexdigest()[:16]
                    documents_ids.append(
                        f"{collection_name}_{batch_start + i}_{content_hash}"
                    )

                # Adiciona o batch ao Chroma
                collection.add(
                    documents=documents_texts,
                    metadatas=documents_metadatas,
                    ids=documents_ids,
                )

                total_added += len(batch_documents)
                print(f"   ✅ Batch adicionado: {len(batch_documents)} documentos")

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
