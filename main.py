"""
FastAPI 메인 서버 + 스케줄러
uvicorn main:app --reload
"""
import logging

import uvicorn
from fastapi import FastAPI, BackgroundTasks
from contextlib import asynccontextmanager

from backend.scheduler.config import task_scheduler
from backend.scheduler.tasks import run_research_task, run_data_pipeline_task, run_email_task

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 실행되는 이벤트"""
    # 시작 시
    logger.info("🚀 서버 시작: 스케줄러 초기화 중...")
    task_scheduler.start()
    
    yield  # 서버 실행
    
    # 종료 시
    logger.info("🛑 서버 종료: 스케줄러 정리 중...")
    task_scheduler.shutdown()


# FastAPI 앱 생성
app = FastAPI(
    title="Kalantir AI Scheduler API",
    description="자동 실행 스케줄러 + API 서버",
    version="1.0.0",
    lifespan=lifespan
)


# ==================== API 엔드포인트 ====================

@app.get("/")
def root():
    """헬스체크"""
    return {
        "status": "running",
        "service": "Kalantir AI Scheduler",
        "scheduler_status": "active" if task_scheduler.is_running() else "inactive"
    }


@app.get("/health")
def health_check():
    """상세 헬스체크"""
    return {
        "status": "healthy",
        "scheduler_running": task_scheduler.is_running(),
        "active_jobs": len(task_scheduler.get_jobs())
    }


@app.get("/schedules")
def get_schedules():
    """등록된 스케줄 조회"""
    jobs = task_scheduler.get_jobs()
    return {
        "total": len(jobs),
        "schedules": [
            {
                "id": job.id,
                "name": job.name,
                "next_run": str(job.next_run_time) if job.next_run_time else None,
                "trigger": str(job.trigger)
            }
            for job in jobs
        ]
    }


@app.post("/trigger/research")
def trigger_research(background_tasks: BackgroundTasks):
    """리서치 작업 수동 실행"""
    background_tasks.add_task(run_research_task)
    return {
        "status": "triggered",
        "task": "research",
        "message": "리서치 작업이 백그라운드에서 시작되었습니다"
    }


@app.post("/trigger/pipeline")
def trigger_pipeline(background_tasks: BackgroundTasks):
    """데이터 파이프라인 수동 실행"""
    background_tasks.add_task(run_data_pipeline_task)
    return {
        "status": "triggered",
        "task": "pipeline",
        "message": "데이터 파이프라인이 백그라운드에서 시작되었습니다"
    }


@app.post("/trigger/email")
def trigger_email(background_tasks: BackgroundTasks):
    """이메일 전송 수동 실행"""
    background_tasks.add_task(run_email_task)
    return {
        "status": "triggered",
        "task": "email",
        "message": "이메일 전송이 백그라운드에서 시작되었습니다"
    }


if __name__ == "__main__":
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )