import os
from dotenv import load_dotenv
import smtplib

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
from pathlib import Path
from sqlalchemy import text

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage
from langchain_core.prompts import load_prompt
from langchain_community.utilities import SQLDatabase

from util.path import PROMPT_DIR

load_dotenv()

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.3)

# 환경변수 설정 (실제 사용 시 설정 필요)
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = os.getenv("GMAIL_SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("GMAIL_EMAIL_KEY")
RECIPIENT_EMAIL = "jeonminwoo0625@gmail.com"

def get_today_report_file():
    """오늘 생성된 리포트 파일 경로를 DB에서 조회"""
    try:
        DB_URI = "postgresql+psycopg2://postgres:1234@localhost:5433/kalantir"
    
        db = SQLDatabase.from_uri(
            DB_URI,
            engine_args={
                "connect_args": {
                    "options": "-c client_encoding=UTF8 -c search_path=ai"
                }
            }
        )
        
        query = text("""
        SELECT file_path, contents, report_type 
        FROM reports 
        WHERE DATE(created_at) = CURRENT_DATE
        ORDER BY created_at DESC 
        LIMIT 1
        """)
        
        with db._engine.connect() as conn:
            result = conn.execute(query).fetchone()
        
        if result:
            print(f"DB 조회 결과: {result[0]}")
            return {
                "file_path": result[0],
                "contents": result[1],
                "report_type": result[2]
            }
        
        print("오늘 생성된 리포트가 없습니다.")
        return None
        
    except Exception as e:
        print(f"DB 조회 오류: {e}")
        return None

def search_web_info(query="삼성전자의 오늘 AI 주요 뉴스"):

    tavily_search = TavilySearch(max_results=3)
    search_docs = tavily_search.invoke(query)
    
    email_search_template = load_prompt(PROMPT_DIR / "search_instructions_email.yaml", encoding="utf-8")
    email_search = email_search_template.format(search_docs=search_docs)
    
    news = llm.invoke([HumanMessage(content=email_search)])
    
    return news.content

def send_email_with_pdf():
    """이메일 전송 함수"""
    try:
        # 1. DB에서 오늘 리포트 파일 정보 가져오기
        report_info = get_today_report_file()
        
        if not report_info:
            print("오늘 생성된 리포트가 없습니다.")
            return
        
        file_path = report_info["file_path"]
        
        # 파일 존재 확인
        if not Path(file_path).exists():
            print(f"파일을 찾을 수 없습니다: {file_path}")
            return
        
        # 2. 웹 검색 수행
        search_results = search_web_info("삼성전자의 오늘 AI 주요 뉴스")
        
        # 3. 이메일 메시지 생성
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECIPIENT_EMAIL
        msg['Subject'] = f"칼란티어 AI 리포트 - {datetime.now().strftime('%Y년 %m월 %d일')}"
        
        # 이메일 본문
        body = f"""
안녕하세요.
칼란티어 AI 비즈니스 팀입니다.

오늘의 삼성전자 AI 주요 뉴스와 
Daily Report(PDF) 첨부하여 보내드립니다.

{search_results}

리포트 타입 : {report_info['report_type']}
생성 시간 : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

감사합니다.
        """
        
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        
        # 4. PDF 파일 첨부
        filename = Path(file_path).name
        with open(file_path, "rb") as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename= {filename}'
            )
            msg.attach(part)
        
        # 5. 이메일 전송
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        
        print(f"이메일 전송 완료: {datetime.now()}")
        
    except Exception as e:
        print(f"이메일 전송 오류: {e}")
        import traceback
        traceback.print_exc()

def job():
    print(f"메일 전송 작업 시작: {datetime.now()}")
    send_email_with_pdf()

if __name__ == "__main__":
    job()