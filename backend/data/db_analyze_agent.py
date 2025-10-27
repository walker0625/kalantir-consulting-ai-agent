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
        llm_model: str = "gpt-4o",
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
        llm = ChatOpenAI(model=self.llm_model, temperature=0.3)
        tools = SQLDatabaseToolkit(db=self.db, llm=llm).get_tools()
        
        self.agent = initialize_agent(
            llm=llm,
            tools=tools,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
        )
        print("SQL Agent 초기화 완료")
        
    def analyze_and_generate_proposals(self) -> str:
        """DB 분석 후 AI 도입 제안 생성"""
        prompt = """
# Role
당신은 PostgreSQL 기반 DB 구조와 데이터를 잘 이해하고,
SQL 쿼리를 통해 인사이트를 도출할 수 있는 DBA이자 사업분석 및 AI 전략 전문가입니다.

# Goal
삼성의 사업 영역에서 AI를 도입하면 효율성과 경쟁력이 높아질 수 있는 3가지 구체적인 제안을 생성하세요.

# Todo
1. DB의 테이블명과 컬럼명을 조회하여, 사업 성과나 경쟁력, 품질, 생산성과 관련된 데이터를 가진 테이블을 식별하라.
2. 관련 테이블을 대상으로 실제 데이터를 **SELECT 쿼리로 조회**하라.  
    - 예: `SELECT * FROM table_name LIMIT 5;`  
    - 수치형 컬럼(`sales`, `revenue`, `utilization`, `metric_value` 등)은 평균/비율/변동성을 계산하라.  
    - 텍스트 컬럼(`note`, `reason`, `description` 등)은 주요 키워드나 패턴을 요약하라.
3. 조회한 데이터를 바탕으로 구체적인 분석 결과를 생성하라.  
    - 단순 스키마 설명이 아니라, **실제 데이터 분석 기반 인사이트**를 도출해야 한다.
4. 마지막으로, 분석 결과를 종합하여 AI를 도입할 수 있는 3가지 제안을 작성하라.
5. 모든 결과는 **한글로** 출력
6. 불필요한 말 없이 간결하게 출력 (예 : 1. 내용 2. 내용 3. 내용)

# Hard Constraint
1. 한번의 Thought가 생성되면 그대로 출력하고 또 다른 Action이 없다고 해서 다시 실행하지 말 것!
2. 마지막 final answer 부분만 출력!
"""
        
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
        chunk_size: int = 300,
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
        llm_model="gpt-4o",
        embedding_model="text-embedding-3-small"
    )
    
    analyzer.run()

if __name__ == "__main__":
    main()