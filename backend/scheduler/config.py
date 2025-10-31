"""
APScheduler 설정 및 스케줄 등록
"""
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import logging

from backend.scheduler.tasks import run_data_pipeline_task, run_research_task, run_email_task, run_pipeline_then_research

logger = logging.getLogger(__name__)


class TaskScheduler:
    """스케줄러 관리 클래스"""
    
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self._setup_schedules()
    
    def _setup_schedules(self):
        # 매일 오전 6시에 데이터 수집 → 리서치 순차 실행
        self.scheduler.add_job(
            run_pipeline_then_research,
            CronTrigger(hour=6, minute=0),
            id='pipeline_and_research',
            name='Pipeline → Research',
            replace_existing=True
        )

        # 매일 오전 9시에 이메일 전송
        self.scheduler.add_job(
            run_email_task,
            CronTrigger(hour=9, minute=0),
            id='email_daily',
            name='Email Sender',
            replace_existing=True
        )
    
    def print_schedules(self):
        """등록된 스케줄 출력"""
        jobs = self.scheduler.get_jobs()
        logger.info("=" * 70)
        logger.info("📅 등록된 스케줄:")
        for job in jobs:
            logger.info(f"  • {job.name} ({job.id})")
            logger.info(f"    ⏰ {job.trigger}")
            if job.next_run_time:
                logger.info(f"    ⏭️  다음 실행: {job.next_run_time}")
        logger.info("=" * 70)
    
    def start(self):
        """스케줄러 시작"""
        self.scheduler.start()
        self.print_schedules()
        logger.info("🚀 스케줄러 시작됨")
    
    def shutdown(self):
        """스케줄러 종료"""
        self.scheduler.shutdown()
        logger.info("🛑 스케줄러 종료됨")
    
    def get_jobs(self):
        """작업 목록 반환"""
        return self.scheduler.get_jobs()
    
    def is_running(self):
        """실행 상태 확인"""
        return self.scheduler.running

# 싱글톤 인스턴스
task_scheduler = TaskScheduler()