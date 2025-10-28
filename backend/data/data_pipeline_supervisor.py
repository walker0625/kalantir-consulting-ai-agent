import logging
from datetime import datetime
from typing import Literal

from db_analyze_agent import DbAnalyzeAgent
from pdf_vectorizer import PdfVectorizer
from web_search_agent import WebSearchAgent

# ----------------------------
# 기본 로깅 설정
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("supervisor.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

class DataPipelineSupervisor:
    """
    PDF / WEB / DB Agent를 중앙에서 제어하는 Supervisor
    - 순차 실행, 예외 처리, 로깅 관리
    - LLM은 각 Agent 내부에서만 사용 (Supervisor는 orchestration 전용)
    """

    def __init__(
        self,
        db_uri: str,
        qdrant_url: str,
        pdf_folder: str
    ):
        self.db_uri = db_uri
        self.qdrant_url = qdrant_url
        self.pdf_folder = pdf_folder
        self.results = {}

    # ----------------------------------------------------------------
    # 개별 실행 래퍼
    # ----------------------------------------------------------------
    def run_db_agent(self):
        logging.info("[DB Agent] 내부 데이터 분석 시작")
        try:
            db_agent = DbAnalyzeAgent(
                db_uri=self.db_uri,
                db_schema="ai",
                collection_name="samsung_internal_db",
                qdrant_url=self.qdrant_url,
                llm_model="gpt-4o"
            )
            result = db_agent.run()
            self.results["db"] = result
            logging.info("[DB Agent] 완료")
        except Exception as e:
            logging.error(f"[DB Agent] 오류 발생: {e}")

    def run_pdf_agent(self):
        logging.info("[PDF Agent] 외부 문서 벡터화 시작")
        try:
            pdf_agent = PdfVectorizer(
                folder_path=self.pdf_folder,
                qdrant_url=self.qdrant_url,
                collection_name="samsung_external_pdf"
            )
            pdf_agent.run()
            self.results["pdf"] = "PDF 처리 완료"
            logging.info("[PDF Agent] 완료")
        except Exception as e:
            logging.error(f"[PDF Agent] 오류 발생: {e}")

    def run_web_agent(self, query: str):
        logging.info("🌐 [Web Agent] 웹 검색 및 임베딩 시작")
        try:
            web_agent = WebSearchAgent(
                query=query,
                qdrant_url=self.qdrant_url,
                collection_name="samsung_external_web"
            )
            web_agent.run()
            self.results["web"] = f"'{query}' 검색 완료"
            logging.info("✅ [Web Agent] 완료")
        except Exception as e:
            logging.error(f"[Web Agent] 오류 발생: {e}")

    # ----------------------------------------------------------------
    # 전체 실행 관리
    # ----------------------------------------------------------------
    def run_all(self, query: str):
        logging.info("="*80)
        logging.info("데이터 파이프라인 Supervisor 시작")
        start_time = datetime.now()

        self.run_db_agent()
        self.run_web_agent(query)
        self.run_pdf_agent()

        duration = datetime.now() - start_time
        logging.info(f"전체 파이프라인 완료 (총 소요 시간: {duration})")
        logging.info("="*80)

        return self.results

    # ----------------------------------------------------------------
    # 단일 실행 모드 (API 호출 등에서 사용)
    # ----------------------------------------------------------------
    def run_mode(self, mode: Literal["db", "web", "pdf", "all"], query: str = ""):
        if mode == "db":
            self.run_db_agent()
        elif mode == "web":
            self.run_web_agent(query)
        elif mode == "pdf":
            self.run_pdf_agent()
        elif mode == "all":
            return self.run_all(query)
        else:
            logging.warning(f"알 수 없는 모드: {mode}")

# ----------------------------
# 단독 실행 예시
# ----------------------------
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    POSTGRES_URL = os.getenv("POSTGRES_URL")
    QDRANT_URL = os.getenv("QDRANT_URL")
    PDF_FOLDER = "../../raw_data/pdf"

    supervisor = DataPipelineSupervisor(
        db_uri=POSTGRES_URL,
        qdrant_url=QDRANT_URL,
        pdf_folder=PDF_FOLDER
    )

    # 전체 파이프라인 실행
    results = supervisor.run_all(query="삼성전자의 AI 관련 최근 사업 전략")
    print("\n[최종 결과 요약]")
    for k, v in results.items():
        print(f"- {k}: {v}")
