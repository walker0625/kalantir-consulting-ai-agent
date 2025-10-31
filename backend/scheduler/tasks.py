"""
스케줄러가 실행할 작업 함수들
"""
import os
import logging
from datetime import datetime

from backend.data.data_pipeline_supervisor import DataPipelineSupervisor
from backend.analysis.discussion_agent import main as research_main
from backend.email.mail_sender import job as email_job

from util.path import PDF_DIR

logger = logging.getLogger(__name__)

def run_data_pipeline_task():
    """데이터 파이프라인 실행"""
    try:
        logger.info(f"[{datetime.now()}] 데이터 파이프라인 시작")

        supervisor = DataPipelineSupervisor(
            db_uri=os.getenv("POSTGRES_URL"),
            qdrant_url=os.getenv("QDRANT_URL"),
            pdf_folder=PDF_DIR
        )
        supervisor.run_all()
        
        logger.info(f"[{datetime.now()}] 데이터 파이프라인 완료")
    except Exception as e:
        logger.error(f"데이터 파이프라인 실패: {e}", exc_info=True)

def run_research_task():
    """리서치 에이전트 실행"""
    try:
        logger.info(f"[{datetime.now()}] 리서치 작업 시작")
        
        research_main()
        
        logger.info(f"[{datetime.now()}] 리서치 작업 완료")
    except Exception as e:
        logger.error(f"리서치 작업 실패: {e}", exc_info=True)

def run_email_task():
    """이메일 전송 실행"""
    try:
        logger.info(f"[{datetime.now()}] 이메일 전송 시작")
        
        email_job()
        
        logger.info(f"[{datetime.now()}] 이메일 전송 완료")
    except Exception as e:
        logger.error(f"이메일 전송 실패: {e}", exc_info=True)
        
def run_pipeline_then_research():
    """데이터 파이프라인 → 리서치 (순차 실행)"""
    logger.info("=" * 70)
    logger.info("📊 통합 작업 시작: 데이터 수집 → 리서치 분석")
    logger.info("=" * 70)
    
    # Step 1: 데이터 수집
    try:
        logger.info("1️⃣ 데이터 파이프라인 시작")
        run_data_pipeline_task()
        logger.info("✅ 데이터 수집 완료")
    except Exception as e:
        logger.error(f"❌ 데이터 수집 실패: {e}")
        logger.warning("⚠️ 리서치를 건너뜁니다")
        return
    
    # Step 2: 리서치 분석
    try:
        logger.info("2️⃣ 리서치 분석 시작")
        run_research_task()
        logger.info("✅ 리서치 분석 완료")
    except Exception as e:
        logger.error(f"❌ 리서치 분석 실패: {e}")
    
    logger.info("=" * 70)
    logger.info("✅ 통합 작업 완료")
    logger.info("=" * 70)