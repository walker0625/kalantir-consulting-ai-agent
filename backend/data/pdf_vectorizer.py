import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant

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

    def load_pdfs(self, pdf_files):
        """PDF 파일들을 로드"""
        all_docs = []
        for path in pdf_files:
            try:
                loader = PDFPlumberLoader(path)
                docs = loader.load()
                
                print(f"Loaded {len(docs)} pages from {os.path.basename(path)}")
                all_docs.extend(docs)
            except Exception as e:
                print(f"{os.path.basename(path)} 로드 실패: {e}")
        print(f"총 {len(all_docs)} pages 로드 완료 (파일 {len(pdf_files)}개).")
        
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

    def run(self):
        """폴더 내 PDF 파일 전체를 처리"""
        pdf_files = self.find_pdf_files()
        
        if not pdf_files:
            return

        docs = self.load_pdfs(pdf_files)
        
        if not docs:
            print("로드된 PDF 문서가 없습니다.")
            return

        split_docs = self.split_documents(docs)
        self.store_in_qdrant(split_docs)

if __name__ == "__main__":

    pdf_vectorizer = PdfVectorizer(
        folder_path="../../raw_data/pdf",
        chunk_size=1000,
        chunk_overlap=100,
        qdrant_url="http://localhost:6333",
        collection_name="samsung_external_pdf",
        embedding_model="text-embedding-3-small"
    )
    pdf_vectorizer.run()
