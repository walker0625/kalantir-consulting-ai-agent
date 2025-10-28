"""
Discussion Agent - LangGraph 기반 컨설턴트 인터뷰 및 보고서 생성 시스템

이 모듈은 여러 전문가(컨설턴트) 페르소나를 활용하여 주제에 대한 심층 인터뷰를 수행하고,
수집된 정보를 바탕으로 종합 보고서를 생성하는 AI 에이전트 시스템입니다.
"""
import operator
from typing import Annotated, List, Callable

from util.consultants import Consultant, CONSULTANT_PROFILES
from util.path import PROMPT_DIR

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage, get_buffer_string
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import load_prompt
from langchain_tavily import TavilySearch
from langchain_community.vectorstores import Qdrant
from langchain_community.utilities import SQLDatabase

from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver

from qdrant_client import QdrantClient
from sqlalchemy import text

load_dotenv()

# LLM 초기화
llm = ChatOpenAI(model='gpt-4o', temperature=0)

class SearchQuery(BaseModel):
    """검색 쿼리 모델"""
    search_query: str = Field(None, description="Search query for retrieval.")

def debug_reducer(old, new):
    """State update 동기화 error 해결용 reducer"""
    return new

class InterviewState(MessagesState):
    """인터뷰 상태"""
    max_num_turns: int
    context: Annotated[list, operator.add]
    consultant: Consultant
    interview: str
    sections: list
    topic: Annotated[str, debug_reducer]

class ResearchGraphState(MessagesState):
    """전체 리서치 그래프 상태"""
    topic: Annotated[str, debug_reducer]
    max_consultants: int
    human_consultant_feedback: str
    consultants: List[Consultant]
    sections: Annotated[list, operator.add]
    introduction: str
    content: str
    conclusion: str
    final_report: str

# ================================
# 인터뷰 노드 함수들
# ================================
def ask_question(state: InterviewState):
    """컨설턴트가 질문을 생성하는 노드"""
    topic = state['topic']
    consultant = state['consultant']
    messages = state['messages']
    
    question_instructions = load_prompt(PROMPT_DIR / "question_instructions.yaml", encoding="utf-8")
    
    system_message = question_instructions.format(topic=topic, goals=consultant.persona)
    question = llm.invoke([SystemMessage(content=system_message)] + messages)
    
    return {'messages': [question]}

def search_web(state: InterviewState):
    
    consultant= state['consultant']
    topic=state['topic']
    
    """웹 검색으로 문서를 찾는 노드"""
    search_instructions = load_prompt(PROMPT_DIR / "search_instructions.yaml", encoding="utf-8").format(
        name=consultant.name, role=consultant.role, 
        department=consultant.department, description=consultant.description, 
        topic=topic, domain_keywords=consultant.domain_keywords, 
        search_focus=consultant.search_focus)
    
    # 응답 형식 지정
    structured_llm = llm.with_structured_output(SearchQuery)
    # LLM으로 검색 쿼리 최적화
    search_query = structured_llm.invoke([SystemMessage(search_instructions)] + state['messages'])
    
    tavily_search = TavilySearch(max_results=6)
    
    print('검색쿼리')
    print(search_query.search_query)
    search_docs = tavily_search.invoke(search_query.search_query)
    print('검색결과')
    print(search_docs)
    
    return {"context": [search_docs]}

def create_retriever(collection_name: str, k: int = 20):
    """Qdrant에서 retriever 생성"""
    QDRANT_URL = 'http://localhost:6333'
    
    client = QdrantClient(url=QDRANT_URL)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vectorstore = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings
    )
    
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

def search_rag(state: InterviewState):
    """RAG를 통해 내부 문서를 검색하는 노드"""
    COLLECTIONS = ['samsung_internal_db', 'samsung_external_web', 'samsung_external_pdf']
    retrievers = {name: create_retriever(name) for name in COLLECTIONS}
    
    query = state['topic']
    
    for name, retriever in retrievers.items():
        results = retriever.invoke(query)
        print(f"\n{name} results:")
        for doc in results[:2]:
            print("-", doc.metadata.get("source"), ":", doc.page_content[:100])
            
    return {"context": [results]}

def answer_question(state: InterviewState):
    """전문가가 답변을 생성하는 노드"""
    consultant = state['consultant']
    messages = state['messages']
    context = state.get('context', [])
    
    answer_instructions = load_prompt(PROMPT_DIR / "answer_instructions.yaml", encoding="utf-8")
    
    system_message = answer_instructions.format(
        goals=consultant.persona,
        context=context
    )
    answer = llm.invoke([SystemMessage(content=system_message)] + messages)
    
    return {'messages': [answer]}

def save_interview(state: InterviewState):
    """인터뷰 내용을 저장하는 노드"""
    messages = state['messages']
    interview = get_buffer_string(messages)
    
    return {'interview': interview}

def write_section(state: InterviewState):
    """인터뷰 내용을 바탕으로 섹션을 작성하는 노드"""
    interview = state['interview']
    context = state['context']
    consultant = state['consultant']
    
    # TODO interview 내용 확인하여 활용 결정
    print('***********컨텍스트시작************')
    print(context)
    print('***********컨텍스트 끝************')
    print('**********인터뷰 시작*************')
    print(interview)
    print('**********인터뷰 끝*************')
    
    section_writer_instructions = load_prompt(PROMPT_DIR / "section_writer_instructions.yaml", encoding="utf-8")
    
    system_message = section_writer_instructions.format(focus=consultant.persona)
    section = llm.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content=f"Use this source material to create a section: {context}")
    ])
    
    return {'sections': [section.content]}

def route_messages(state: InterviewState):
    """대화 종료 여부를 판단하는 라우팅 함수"""
    messages = state['messages']
    max_num_turns = state.get('max_num_turns', 3)
    num_responses = len([m for m in messages if isinstance(m, AIMessage)])
    
    if num_responses >= max_num_turns:
        return 'save_interview'
    
    last_question = messages[-2]
    
    if "Thank you so much for your help" in last_question.content:
        return 'save_interview'
    
    return 'ask_question'

# ================================
# 인터뷰 그래프 구성
# ================================
def build_interview_graph():
    """인터뷰 그래프를 구성하고 반환"""
    interview_builder = StateGraph(InterviewState)
    
    # 노드 추가
    interview_builder.add_node('ask_question', ask_question)
    interview_builder.add_node('search_web', search_web)
    interview_builder.add_node('search_rag', search_rag)
    interview_builder.add_node('answer_question', answer_question)
    interview_builder.add_node('save_interview', save_interview)
    interview_builder.add_node('write_section', write_section)
    
    # 엣지 연결
    interview_builder.add_edge(START, 'ask_question')
    interview_builder.add_edge('ask_question', 'search_web')
    interview_builder.add_edge('ask_question', 'search_rag')
    interview_builder.add_edge('search_web', 'answer_question')
    interview_builder.add_edge('search_rag', 'answer_question')
    interview_builder.add_conditional_edges(
        'answer_question', route_messages, ['ask_question', 'save_interview']
    )
    interview_builder.add_edge('save_interview', 'write_section')
    interview_builder.add_edge('write_section', END)
    
    memory = MemorySaver()
    return interview_builder.compile(checkpointer=memory).with_config(
        run_name='Conduct Interviews'
    )


# ================================
# 메인 리서치 그래프 노드 함수들
# ================================

def setting_consultants(state: ResearchGraphState):
    # 최대 컨설턴트 수 가져오기
    max_consultants = state.get("max_consultants", 3)
    
    # 프로필을 Consultant 객체로 변환
    consultants = [
        Consultant(**profile) 
        for profile in CONSULTANT_PROFILES[:max_consultants]
    ]
    return {'consultants': consultants}

# TODO 병렬실행을 위해 추가한 더미 함수 - 불필요시 제거
# 노드 함수 (상태 업데이트만)
def initiate_all_interviews_node(state: ResearchGraphState):
    # 상태는 그대로 반환 (pass-through) 
    return {}

def route_to_interviews(state: ResearchGraphState):
    """인터뷰 라우팅 함수"""
    max_consultants = state.get('max_consultants', 3)
    
    return [
        Send("conduct_interview", {
            "consultant": consultant,
            "messages": [
                HumanMessage(content="Welcome! We'll begin the interview now.")
            ],
            "max_num_turns": 3,
            "context": [],
            "topic": state['topic'],
            "sections": state['sections']
        })
        for consultant in state['consultants'][:max_consultants]
    ]

def write_report(state: ResearchGraphState):
    """메인 보고서를 작성하는 노드"""
    sections = state['sections']
    topic = state['topic']
    
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    
    report_writer_instructions = load_prompt(PROMPT_DIR / "report_writer_instructions.yaml", encoding="utf-8")
    
    system_message = report_writer_instructions.format(
        topic=topic,
        context=formatted_str_sections
    )
    report = llm.invoke([SystemMessage(content=system_message)])
    
    return {'content': report.content}

def write_introduction(state: ResearchGraphState):
    """인트로를 작성하는 노드"""
    topic = state['topic']
    sections = state['sections']
    
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    
    intro_conclusion_instructions = load_prompt(PROMPT_DIR / "intro_conclusion_instructions.yaml", encoding="utf-8")
    
    instructions = intro_conclusion_instructions.format(
        topic=topic,
        formatted_str_sections=formatted_str_sections
    )
    intro = llm.invoke([
        SystemMessage(content=instructions),
        HumanMessage(content=f"Here is the main body:\n\n{formatted_str_sections}")
    ])
    
    return {'introduction': intro.content}

def write_conclusion(state: ResearchGraphState):
    """결론을 작성하는 노드"""
    topic = state['topic']
    sections = state['sections']
    
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
    
    intro_conclusion_instructions = load_prompt(PROMPT_DIR / "intro_conclusion_instructions.yaml", encoding="utf-8")
    
    instructions = intro_conclusion_instructions.format(
        topic=topic,
        formatted_str_sections=formatted_str_sections
    )
    conclusion = llm.invoke([
        SystemMessage(content=instructions),
        HumanMessage(content=f"Here is the main body:\n\n{formatted_str_sections}")
    ])
    
    return {'conclusion': conclusion.content}

def finalize_report(state: ResearchGraphState):
    """최종 보고서를 완성하는 노드"""
    introduction = state['introduction']
    content = state['content']
    conclusion = state['conclusion']
    topic = state['topic']
    
    final_report = f"# {topic}\n\n{introduction}\n\n---\n\n{content}\n\n---\n\n{conclusion}"
    
    return {'final_report': final_report}


# ================================
# 메인 리서치 그래프 구성
# ================================
def build_research_graph():
    """메인 리서치 그래프를 구성하고 반환"""
    # 인터뷰 그래프 먼저 빌드
    interview_graph = build_interview_graph()
    
    # 메인 그래프 생성
    builder = StateGraph(ResearchGraphState)
    
    # 노드 추가
    builder.add_node("setting_consultants", setting_consultants)
    builder.add_node("initiate_all_interviews", initiate_all_interviews_node)
    builder.add_node("conduct_interview", interview_graph)
    builder.add_node("write_report", write_report)
    builder.add_node("write_introduction", write_introduction)
    builder.add_node("write_conclusion", write_conclusion)
    builder.add_node("finalize_report", finalize_report)
    
    # 엣지 연결
    builder.add_edge(START, "setting_consultants")
    builder.add_edge("setting_consultants", "initiate_all_interviews")
    builder.add_conditional_edges(
        "initiate_all_interviews",
        route_to_interviews,
        ["conduct_interview"]
    )
    builder.add_edge("conduct_interview", "write_report")
    builder.add_edge("conduct_interview", "write_introduction")
    builder.add_edge("conduct_interview", "write_conclusion")
    builder.add_edge(
        ["write_conclusion", "write_report", "write_introduction"], "finalize_report"
    )
    builder.add_edge("finalize_report", END)
    
    # 컴파일
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


# ================================
# 9. 유틸리티 함수들
# ================================

def invoke_graph(
    graph: CompiledStateGraph,
    inputs: dict,
    config: RunnableConfig,
    node_names: List[str] = [],
    callback: Callable = None,
):
    """
    LangGraph 앱의 실행 결과를 스트리밍하여 출력하는 함수
    
    Args:
        graph: 실행할 컴파일된 LangGraph 객체
        inputs: 그래프에 전달할 입력값
        config: 실행 설정
        node_names: 출력할 노드 이름 목록
        callback: 각 청크 처리를 위한 콜백 함수
    """
    def format_namespace(namespace):
        return namespace[-1].split(":")[0] if len(namespace) > 0 else "root graph"
    
    for namespace, chunk in graph.stream(
        inputs, config, stream_mode="updates", subgraphs=True
    ):
        for node_name, node_chunk in chunk.items():
            if len(node_names) > 0 and node_name not in node_names:
                continue
            
            if callback is not None:
                callback({"node": node_name, "content": node_chunk})
            else:
                print("\n" + "=" * 50)
                formatted_namespace = format_namespace(namespace)
                if formatted_namespace == "root graph":
                    print(f"🔄 Node: \033[1;36m{node_name}\033[0m 🔄")
                else:
                    print(
                        f"🔄 Node: \033[1;36m{node_name}\033[0m in [\033[1;33m{formatted_namespace}\033[0m] 🔄"
                    )
                print("- " * 25)
                
                if isinstance(node_chunk, dict):
                    for k, v in node_chunk.items():
                        if isinstance(v, BaseMessage):
                            v.pretty_print()
                        elif isinstance(v, list):
                            for list_item in v:
                                if isinstance(list_item, BaseMessage):
                                    list_item.pretty_print()
                                else:
                                    print(list_item)
                        else:
                            print(f"\033[1;32m{k}\033[0m:\n{v}")
                print("=" * 50)


def save_report_to_db(report: str, report_type: str = "daily"):
    """보고서를 데이터베이스에 저장하는 함수"""
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
    INSERT INTO reports (contents, report_type)
    VALUES (:final_report, :report_type)
    RETURNING id;
    """)
    
    params = {"final_report": report, "report_type": report_type}
    
    with db._engine.connect() as conn:
        result = conn.execute(query, params)
        new_id = result.scalar()
        conn.commit()
        
    return new_id


# ================================
# 메인 실행 함수
# ================================

def main():
    """메인 실행 함수"""
    # 리서치 그래프 빌드
    graph = build_research_graph()
    
    # 실행 설정
    config = {"configurable": {"thread_id": "1"}}
    
    # 초기 입력
    initial_inputs = {
        "topic": "AI Integration: How can Samsung Electronics integrate AI to improve productivity and enhance its long-term intrinsic value?",
        "max_consultants": 3,
        "consultants": [],
        "sections": [],
        "introduction": "",
        "content": "",
        "conclusion": "",
        "final_report": ""
    }
    
    # 그래프 실행
    print("=" * 80)
    print("Starting Research Process...")
    print("=" * 80)
    
    invoke_graph(graph, initial_inputs, config)
    
    # 최종 보고서 가져오기
    final_state = graph.get_state(config)
    final_report = final_state.values.get("final_report")
    
    # 보고서 출력
    print("\n" + "=" * 80)
    print("FINAL REPORT")
    print("=" * 80)
    print(final_report)
    
    # 보고서 저장 (옵션)
    report_id = save_report_to_db(final_report, "daily")
    print(f"\nReport saved to database with ID: {report_id}")
    
    return final_report


if __name__ == "__main__":
    main()