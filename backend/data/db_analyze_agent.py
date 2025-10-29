import os
from dotenv import load_dotenv
from datetime import datetime

from langchain.agents import initialize_agent, AgentType
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_community.vectorstores import Qdrant
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.prompts import load_prompt

from util.path import PROMPT_DIR

load_dotenv()

class DbAnalyzeAgent:
    """
    고객사 내부 RDB 조회 및 분석 -> 서비스 벡터 DB에 분석 결과 저장
    (SQL Agent를 통해 DB 분석 후, 결과를 Qdrant에 저장)
    """
    
    def __init__(
        self,
        db_uri: str,
        db_schema: str = "ai",
        collection_name: str = "samsung_internal_db",
        qdrant_url: str = "http://localhost:6333",
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Args:
            db_uri: PostgreSQL 연결 URI
            db_schema: DB 스키마명
            qdrant_url: Qdrant 서버 URL
            llm_model: 사용할 LLM 모델명
            embedding_model: 사용할 임베딩 모델명
        """
        self.db_uri = db_uri
        self.db_schema = db_schema
        self.collection_name = collection_name
        self.qdrant_url = qdrant_url
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        
        # 컴포넌트 초기화
        self.db = None
        self.agent = None
        
    def setup_database(self):
        """PostgreSQL DB 연결 설정"""
        self.db = SQLDatabase.from_uri(
            self.db_uri,
            engine_args={
                "connect_args": {
                    "options": f"-c client_encoding=UTF8 -c search_path={self.db_schema}"
                }
            }
        )
        print("PostgreSQL 연결 완료")
        
    def setup_agent(self):
        """SQL Agent 초기화"""
        llm = ChatOpenAI(model=self.llm_model, temperature=0.1)
        tools = SQLDatabaseToolkit(db=self.db, llm=llm).get_tools()
        
        self.agent = initialize_agent(
            llm=llm,
            tools=tools,
            agent_type=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5 # agent 특성상 text 결과의 경우 무한루프 가능성 있으므로 제한 걸어야 함
        )
        print("SQL Agent 초기화 완료")
        
    def analyze_and_generate_proposals(self) -> str:
        """DB 분석 후 AI 도입 제안 생성"""
    
        prompt = load_prompt(PROMPT_DIR / "db_analysis_instructions.yaml", encoding="utf-8")
    
        result = self.agent.invoke({"input": prompt})
        output = result['output']
        
        print("\n" + "="*80)
        print("분석 결과:")
        print("="*80)
        print(output)
        print("="*80 + "\n")
        
        return output
        
    def store_to_qdrant(
        self, 
        text: str, 
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> Qdrant:
        """분석 결과를 Qdrant에 저장"""
        # 텍스트 분할
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_text(text)
        
        # Document 객체 생성
        today = datetime.now().strftime("%Y%m%d")
        docs = [
            Document(
                page_content=chunk,
                metadata={
                    "source": f"samsung_internal_db_{today}", 
                    "chunk_id": i
                }
            )
            for i, chunk in enumerate(chunks)
        ]
        
        print(f"{len(chunks)}개 청크로 분할 완료")
        
        embedding = OpenAIEmbeddings(model=self.embedding_model)
        
        # Qdrant에 저장
        db = Qdrant.from_documents(
            documents=docs,
            embedding=embedding,
            collection_name=self.collection_name,
            url=self.qdrant_url
        )
        
        print(f"Qdrant 컬렉션 '{self.collection_name}'에 저장 완료")
        
        return db
        
    def run(self):
        """전체 파이프라인 실행"""
        print("DB AGENT 분석 시스템 시작\n")
        
        # 1. 초기화
        self.setup_database()
        self.setup_agent()
        
        # 2. 분석 실행
        print("\nDB 분석 중...")
        analysis_result = self.analyze_and_generate_proposals()
        
        # 3. Qdrant에 저장
        print("\nQdrant에 저장 중...")
        self.store_to_qdrant(analysis_result)
        
        print("\n모든 작업 완료!")
        
        return analysis_result

def main():
    """메인 실행 함수"""
    
    POSTGRES_URL = os.getenv("POSTGRES_URL")
    QDRANT_URL = os.getenv("QDRANT_URL")
    
    analyzer = DbAnalyzeAgent(
        db_uri=POSTGRES_URL,
        db_schema="ai",
        collection_name="samsung_internal_db",
        qdrant_url=QDRANT_URL,
        llm_model="gpt-4o-mini",
        embedding_model="text-embedding-3-small"
    )
    
    analyzer.run()

if __name__ == "__main__":
    main()