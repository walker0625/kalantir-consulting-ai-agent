import os
import hashlib
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, PayloadSchemaType

from util.path import PDF_DIR

load_dotenv()

class PdfVectorizer:
    """지정된 폴더 내 모든 pdf 파일을 임베딩 후 Qdrant에 저장하는 클래스"""
    def __init__(
        self,
        folder_path: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        embedding_model: str = "text-embedding-3-small",
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "samsung_external_pdf"
    ):
        self.folder_path = folder_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding = OpenAIEmbeddings(model=embedding_model)
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.qdrant_client = QdrantClient(url=qdrant_url)
        
        # 🔑 초기화 시 인덱스 설정
        self._ensure_index()

    def _ensure_index(self):
        """file_hash 필드에 인덱스 생성 (없으면 생성, 있으면 스킵)"""
        try:
            # 컬렉션 존재 여부 확인(collection 없는데 index 생성 오류 방지)
            collections = self.qdrant_client.get_collections()
        
            if not any(c.name == self.collection_name for c in collections.collections):
                print(f"Collection '{self.collection_name}' 없음 - 첫 실행 저장 이후 _ensure_index 다시 호출 됨")
                return
            
            # 인덱스 생성(기존에 같은 index가 있으면 그냥 넘어가므로, Exception은 발생하지 않음)
            self.qdrant_client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.file_hash",
                field_schema=PayloadSchemaType.KEYWORD
            )
        
        except Exception as e:
            print(f"⚠️ 인덱스 생성 에러: {e}")

    def calculate_file_hash(self, file_path: str) -> str:
        """파일의 SHA256 해시 계산 (청킹 전에 빠르게 계산)"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def is_already_processed(self, file_hash: str) -> bool:
        """Qdrant에 이미 처리된 파일인지 확인"""
        try:
            collections = self.qdrant_client.get_collections()
            collection_exists = any(
                c.name == self.collection_name 
                for c in collections.collections
            )
            
            if not collection_exists:
                return False
            
            result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.file_hash",
                            match=MatchValue(value=file_hash)
                        )
                    ]
                ),
                limit=1,
                with_payload=False,
                with_vectors=False
            )
            
            return len(result[0]) > 0
            
        except Exception as e:
            print(f"  [DEBUG] 에러 발생: {e}")
            return False

    def find_pdf_files(self):
        """폴더 내의 모든 pdf 파일 경로 수집"""
        if not os.path.exists(self.folder_path):
            raise FileNotFoundError(f"폴더를 찾을 수 없습니다: {self.folder_path}")

        pdf_files = [
            os.path.join(self.folder_path, f)
            for f in os.listdir(self.folder_path)
            if f.lower().endswith(".pdf")
        ]

        if not pdf_files:
            print(f"폴더 내에 pdf 파일이 없습니다: {self.folder_path}")
        else:
            print(f"Found {len(pdf_files)} PDF files in folder: {self.folder_path}")

        return pdf_files

    def load_pdfs(self, pdf_files_with_hash):
        """PDF 파일들을 로드 (해시값과 함께)"""
        all_docs = []
        for path, file_hash in pdf_files_with_hash:
            try:
                loader = PDFPlumberLoader(path)
                docs = loader.load()
                
                # 각 문서에 file_hash 메타데이터 추가
                for doc in docs:
                    doc.metadata["file_hash"] = file_hash
                    doc.metadata["file_name"] = os.path.basename(path)
                    doc.metadata["processed_at"] = datetime.now().isoformat()

                print(f"Loaded {len(docs)} pages from {os.path.basename(path)}")
                all_docs.extend(docs)
            except Exception as e:
                print(f"{os.path.basename(path)} 로드 실패: {e}")
        
        print(f"총 {len(all_docs)} pages 로드 완료 (파일 {len(pdf_files_with_hash)}개).")
        return all_docs

    def split_documents(self, docs):
        """문서를 chunk 단위로 분할"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        split_docs = splitter.split_documents(docs)
        print(f"Split into {len(split_docs)} chunks.")

        return split_docs

    def store_in_qdrant(self, docs):
        """Qdrant에 문서 저장"""
        print(f"Saving {len(docs)} chunks to Qdrant collection: {self.collection_name}")

        Qdrant.from_documents(
            documents=docs,
            embedding=self.embedding,
            collection_name=self.collection_name,
            url=self.qdrant_url
        )

        print(f"Successfully stored {len(docs)} docs in '{self.collection_name}'.\n")
        
        # 🔑 저장 후 인덱스 다시 확인 (첫 실행 시 컬렉션이 방금 생성됨)
        self._ensure_index()

    def run(self):
        """폴더 내 PDF 파일 전체를 처리 (중복 제거 포함)"""
        pdf_files = self.find_pdf_files()

        if not pdf_files:
            return

        # 중복 체크: 새 파일만 필터링
        new_files = []
        skipped_count = 0
        
        print("\n=== 중복 체크 시작 ===")
        for pdf_path in pdf_files:
            file_hash = self.calculate_file_hash(pdf_path)
            
            if self.is_already_processed(file_hash):
                print(f"⏭️  Skip: {os.path.basename(pdf_path)} (이미 처리됨)")
                skipped_count += 1
            else:
                print(f"✅ New: {os.path.basename(pdf_path)}")
                new_files.append((pdf_path, file_hash))
        
        print(f"\n총 {len(pdf_files)}개 파일 중:")
        print(f"  - 새 파일: {len(new_files)}개")
        print(f"  - 스킵: {skipped_count}개")
        print("=" * 40 + "\n")

        if not new_files:
            print("✅ 처리할 새 파일이 없습니다.")
            return

        # 새 파일만 로드/청킹/저장
        docs = self.load_pdfs(new_files)

        if not docs:
            print("로드된 PDF 문서가 없습니다.")
            return

        split_docs = self.split_documents(docs)
        self.store_in_qdrant(split_docs)


if __name__ == "__main__":
    pdf_vectorizer = PdfVectorizer(
        folder_path=PDF_DIR,
        chunk_size=1000,
        chunk_overlap=100,
        qdrant_url="http://localhost:6333",
        collection_name="samsung_external_pdf",
        embedding_model="text-embedding-3-small"
    )
    pdf_vectorizer.run()