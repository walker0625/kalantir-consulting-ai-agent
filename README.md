# 🧠 Kalantir: AI 전략 Auto Mailing 컨설팅

> **LangGraph 기반의 멀티 페르소나 인터뷰 및 데이터 파이프라인 자동화 솔루션**

## 📖 Project Overview

**Kalantir**는 기업의 전략적 의사결정을 지원하기 위해 설계된 **자율형 AI 컨설팅 에이전트**입니다.
단순한 정보 검색을 넘어, AI가 다양한 전문가 페르소나(CEO, CTO, 전략 기획자 등)를 생성하여 심층 인터뷰를 수행하고, 내부 데이터베이스(DB)와 외부 데이터(Web/PDF)를 종합적으로 분석하여 전문적인 컨설팅 리포트를 작성합니다.

이 프로젝트는 **LangGraph**를 활용한 순환형(Cyclic) 그래프 구조로 복잡한 추론 과정을 제어하며, **Supervisor 패턴**을 통해 이기종 데이터 소스를 효율적으로 통합 관리합니다.

### 🎯 Key Features

* **Autonomous Research Pipeline**: 주제 선정부터 자료 조사, 분석, 리포트 작성까지 전 과정을 자동화.
* **Multi-Persona Interviews**: 주제에 적합한 가상의 전문가(AI Consultants)를 생성하고, 병렬 인터뷰를 수행하여 다각도의 인사이트 도출.
* **Hybrid RAG System**: 기업 내부 DB(SQL), 외부 문서(PDF Vector Store), 최신 웹 정보(MCP Search)를 결합한 고도화된 검색 증강 생성.
* **Automated Scheduling**: FastAPI와 백그라운드 스케줄러를 통합하여 정기적인 리서치 및 데이터 파이프라인 실행.

---

<img width="2752" height="1536" alt="system" src="https://github.com/user-attachments/assets/12da02c2-8c26-4929-a28f-55738a56b735" />

## 🏗️ System Architecture

이 시스템은 크게 **Service Layer(FastAPI + Scheduler)**, **Data Pipeline Layer**, **Agentic Analysis Layer(LangGraph)**의 3계층으로 구성되어 있습니다.

### 1. High-Level Architecture

```text
+---------------------------------------------------------------+
|                      User / Client Interface                  |
+---------------------------------------------------------------+
           |                                     ^
           v (REST API)                          | (Report PDF)
+---------------------------------------------------------------+
|                    FastAPI Backend Server                     |
|  +----------------+   +----------------+   +---------------+  |
|  | Task Scheduler |-->| Research Task  |-->| Email Sender  |  |
|  +----------------+   +----------------+   +---------------+  |
+---------------------------------------------------------------+
           |
           v Triggers
+---------------------------------------------------------------+
|                     Core AI Logic Layers                      |
|                                                               |
|  [ Data Pipeline Supervisor ]      [ Analysis Graph Agent ]   |
|   - Web Search MCP Agent            - Persona Setting         |
|   - PDF Vectorizer                  - Parallel Interviews     |
|   - DB Analyze Agent                - Report Synthesis        |
|                                                               |
+---------------------------------------------------------------+
           |                  |                  |
           v                  v                  v
    +-------------+    +-------------+    +-------------+
    | PostgreSQL  |    | Qdrant (DB) |    |  Web / PDF  |
    +-------------+    +-------------+    +-------------+

```

### 2. Analysis Workflow (LangGraph)

리서치 프로세스는 **Hierarchical Graph** 구조를 따릅니다. 상위 그래프(Research Graph)가 전체 흐름을 제어하고, 하위 그래프(Interview Graph)가 각 전문가와의 대화를 수행합니다.

```text
[ Main Research Graph ]
+-------+      +---------------------+      +----------------+
| Start | ---> | Setting Consultants | ---> | Initiate Views |
+-------+      +---------------------+      +----------------+
                                                    |
          +-----------------------------------------+
          | (Dynamic Routing / Parallel Execution)
          v
   +---------------------------------------------------+
   | [ Sub-Graph: Conduct Interview ]                  |
   |                                                   |
   |  Ask Question <--> Answer Question                |
   |       |                   ^                       |
   |       v                   |                       |
   |  Search (Web/RAG) --------+                       |
   |                                                   |
   |  (End condition met?) --> Save & Write Section    |
   +---------------------------------------------------+
          |
          v
   +---------------------+      +------------------+      +-----+
   | Synthesize Report   | ---> | Finalize & Save  | ---> | END |
   +---------------------+      +------------------+      +-----+

```

### 3. Data Pipeline (Supervisor Pattern)

데이터 수집 단계에서는 Supervisor가 각 데이터 소스별 특화 에이전트를 조율하여 에러를 격리하고 실행을 보장합니다.

```text
                          +------------------------+
                          | Data Pipeline Supervisor|
                          +------------------------+
                                      |
          +---------------------------+---------------------------+
          |                           |                           |
          v                           v                           v
+------------------+        +------------------+        +------------------+
| Web Search Agent |        |    PDF Agent     |        |     DB Agent     |
| (MCP Protocol)   |        | (Vectorizer)     |        | (SQL Analyst)    |
+------------------+        +------------------+        +------------------+
          |                           |                           |
          v                           v                           v
  [Search Results]            [Vector Store]              [SQL Insights]

```

---

## 🛠️ Technology Stack

| Category | Technologies |
| --- | --- |
| **LLM Orchestration** | **LangGraph**, **LangChain** |
| **Backend** | **FastAPI**, Uvicorn |
| **Scheduler** | APScheduler (BackgroundTasks integration) |
| **Database** | **PostgreSQL** (Metadata/Reports), **Qdrant** (Vector Store) |
| **Search / RAG** | **MCP** (Model Context Protocol), Custom Web Searcher |
| **Utilities** | PDF Generation (ReportLab/WeasyPrint implied), Dotenv |

---

## 💡 Technical Highlights

### 1. Hierarchical State Graph Architecture

* **문제**: 단일 에이전트로는 복잡한 인터뷰와 리서치 문맥을 유지하기 어려움.
* **해결**: `LangGraph`를 사용하여 상위의 '리서치 관리자'와 하위의 '인터뷰 수행자'로 역할을 분리. `CompiledStateGraph`를 중첩(Subgraph)시켜 각 인터뷰의 상태(`InterviewState`)와 전체 리서치 상태(`ResearchGraphState`)를 독립적이면서도 유기적으로 관리했습니다.

### 2. Robust Data Pipeline Supervisor

* **문제**: 웹 크롤링, DB 연결, PDF 파싱 등 다양한 데이터 소스 접근 시 하나의 실패가 전체 파이프라인을 중단시킬 위험.
* **해결**: **Supervisor Pattern**을 도입하여 `DataPipelineSupervisor` 클래스가 각 에이전트(`Web`, `PDF`, `DB`)의 실행을 캡슐화하고 예외를 중앙에서 제어하도록 설계했습니다. 이를 통해 부분적인 실패가 발생해도 전체 파이프라인의 안전성을 보장합니다.

### 3. Dynamic Persona Generation & Parallel Execution

* **기능**: 사용자의 주제(Topic)에 따라 가장 적합한 전문가 페르소나(예: "AI 윤리 전문가", "시스템 아키텍트")를 동적으로 생성.
* **최적화**: `LangGraph`의 `Send` API와 `map-reduce` 패턴을 활용하여, 여러 전문가와의 인터뷰를 **병렬(Parallel)**로 수행함으로써 리포트 생성 시간을 획기적으로 단축했습니다.

### 4. Background Scheduling Integration

* **구현**: 단순 API 서버를 넘어, `FastAPI`의 `lifespan` 이벤트를 활용해 서버 구동 시 스케줄러를 함께 초기화합니다. 이를 통해 별도의 워커 프로세스 없이도 정기적인 리서치 작업과 데이터 파이프라인 갱신을 수행하는 **Self-Contained Agent System**을 구축했습니다.

---

## 📂 Directory Structure

```bash
kalantir-consulting-ai-agent
├── backend/
│   ├── analysis/
│   │   ├── nodes/             # LangGraph 노드 정의 (인터뷰, 리포트 작성 등)
│   │   ├── retrieval/         # RAG 및 웹 검색 로직
│   │   └── discussion_agent.py # 메인 그래프 조립 (Research & Interview Graph)
│   ├── data/
│   │   ├── data_pipeline_supervisor.py # 데이터 수집 파이프라인 관리자
│   │   └── [web/pdf/db]_agent.py       # 개별 데이터 소스 처리 에이전트
│   ├── scheduler/             # 스케줄러 설정 및 작업 정의
│   └── prompt/                # 에이전트 페르소나 및 지시 프롬프트 (YAML)
├── util/                      # PDF 생성, 경로 설정 등 유틸리티
├── main.py                    # FastAPI 진입점 및 스케줄러 초기화
├── pyproject.toml             # 의존성 관리
└── README.md

```

*** Email로 자동 전송되는 pdf 예시 파일은 /report 폴더 안에 파일로 확인 가능합니다.
