"""
Samsung Data Pipeline
웹 검색 결과와 PDF 문서를 Qdrant 벡터 데이터베이스에 저장하는 파이프라인
"""

import os
from datetime import datetime
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch
from langchain.schema import Document
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from qdrant_client import QdrantClient


class SamsungDataPipeline:
    """삼성 관련 데이터를 수집하고 벡터 DB에 저장하는 파이프라인"""
    
    def __init__(
        self, 
        qdrant_url: str = "http://localhost:6333"
    ):
        """
        Args:
            qdrant_url: Qdrant 서버 URL
        """
        load_dotenv()
        
        self.embedding = OpenAIEmbeddings(model="text-embedding-3-small")
        
        if "LANGSMITH_PROJECT" in os.environ:
            print(f"LangSmith Project: {os.environ['LANGSMITH_PROJECT']}")
    
    def search_web_and_store(self, 
                             query: str, 
                             collection_name: str = "samsung_external_web"):
        """
        웹 검색을 수행하고 결과를 Qdrant에 저장
        
        Args:
            query: 검색 쿼리
            collection_name: Qdrant 컬렉션 이름
        """
        print(f"\n[Web Search] 쿼리: {query}")
        
        # 1. LLM 및 도구 설정
        llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
        tavily_search = TavilySearch(max_results=3)
        
        # 2. 프롬프트 설정
        prompt = ChatPromptTemplate.from_messages([
            ('system', '너는 툴만 사용해서 사실을 확인하고 간단하게 대답해줘'),
            ('human', '{input}'),
            ('placeholder', '{agent_scratchpad}')
        ])
        
        # 3. 에이전트 생성
        agent = create_openai_tools_agent(
            llm=llm,
            tools=[tavily_search],
            prompt=prompt
        )
        
        # 4. 에이전트 실행
        executor = AgentExecutor(
            agent=agent,
            tools=[tavily_search],
            verbose=True
        )
        
        result = executor.invoke({'input': query})
        print(f"\n[Result] {result['output'][:200]}...")
        
        # 5. 결과를 청크로 나누기
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(result["output"])
        
        # 6. Document 객체로 변환
        today = datetime.now().strftime("%Y%m%d")
        docs = [
            Document(
                page_content=chunk,
                metadata={
                    "source": f"samsung_external_web_{today}",
                    "chunk_id": i,
                    "query": query
                }
            )
            for i, chunk in enumerate(chunks)
        ]
        
        # 7. Qdrant에 저장
        db = Qdrant.from_documents(
            documents=docs,
            embedding=self.embedding,
            collection_name=collection_name,
            url=self.qdrant_client._client.rest_uri
        )
        
        print(f"[Success] {len(docs)}개 청크가 '{collection_name}' 컬렉션에 저장되었습니다.")
        return result
    
    def process_pdf_and_store(
        self, 
        pdf_path: str, 
        collection_name: str = "samsung_external_pdf",
        chunk_size: int = 1000,
        chunk_overlap: int = 100
    ):
        """
        PDF 문서를 처리하고 Qdrant에 저장
        
        Args:
            pdf_path: PDF 파일 경로
            collection_name: Qdrant 컬렉션 이름
            chunk_size: 청크 크기
            chunk_overlap: 청크 오버랩
        """
        print(f"\n[PDF Processing] 파일: {pdf_path}")
        
        # 1. PDF 로드
        loader = PDFPlumberLoader(pdf_path)
        docs = loader.load()
        print(f"[Info] 총 {len(docs)} 페이지 로드됨")
        
        # 2. 텍스트 분할
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        split_docs = splitter.split_documents(docs)
        
        # 3. 메타데이터 추가
        today = datetime.now().strftime("%Y%m%d")
        for doc in split_docs:
            doc.metadata["processed_date"] = today
            doc.metadata["source_file"] = os.path.basename(pdf_path)
        
        # 4. Qdrant에 저장
        Qdrant.from_documents(
            documents=split_docs,
            embedding=self.embedding,
            collection_name=collection_name,
            url=self.qdrant_client._client.rest_uri
        )
        
        print(f"[Success] {len(split_docs)}개 청크가 '{collection_name}' 컬렉션에 저장되었습니다.")
        return split_docs


def main():
    """메인 실행 함수"""
    
    # 파이프라인 초기화
    pipeline = SamsungDataPipeline(qdrant_url="http://localhost:6333")
    
    # Qdrant 연결 확인
    pipeline.check_qdrant_connection()
    
    # 1. 웹 검색 및 저장
    query = "삼성전자가 ai를 도입하면 좋을만한 사업에 대해서 자세하게 찾아줘"
    pipeline.search_web_and_store(query)
    
    # 2. PDF 처리 및 저장
    pdf_path = "../../raw_data/Samsung_Electronics_Sustainability_Report_2025_KOR.pdf"
    
    # PDF 파일 존재 확인
    if os.path.exists(pdf_path):
        pipeline.process_pdf_and_store(pdf_path)
    else:
        print(f"[Warning] PDF 파일을 찾을 수 없습니다: {pdf_path}")
        print("PDF 경로를 확인하거나 해당 부분을 주석 처리하세요.")


if __name__ == "__main__":
    main()