"""메인 그래프 조립 및 실행"""
from typing import List, Callable
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_community.utilities import SQLDatabase
from langgraph.graph import START, END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from sqlalchemy import text

from backend.analysis.nodes.states import InterviewState, ResearchGraphState
from backend.analysis.nodes.interview_nodes import ask_question, answer_question, save_interview, write_section, route_messages
from backend.analysis.nodes.report_nodes import setting_consultants, write_report, write_introduction, write_conclusion, finalize_report
from backend.analysis.nodes.routing import initiate_all_interviews_node, route_to_interviews
from backend.analysis.retrieval.rag import search_rag
from backend.analysis.retrieval.web_search import search_web

from util.pdf_maker import make_pdf_and_save

def build_interview_graph():
    """인터뷰 그래프 구성"""
    interview_builder = StateGraph(InterviewState)
    
    interview_builder.add_node('ask_question', ask_question)
    interview_builder.add_node('search_web', search_web)
    interview_builder.add_node('search_rag', search_rag)
    interview_builder.add_node('answer_question', answer_question)
    interview_builder.add_node('save_interview', save_interview)
    interview_builder.add_node('write_section', write_section)
    
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

def build_research_graph():
    """메인 리서치 그래프 구성"""
    interview_graph = build_interview_graph()
    
    builder = StateGraph(ResearchGraphState)
    
    builder.add_node("setting_consultants", setting_consultants)
    builder.add_node("initiate_all_interviews", initiate_all_interviews_node)
    builder.add_node("conduct_interview", interview_graph)
    builder.add_node("write_report", write_report)
    builder.add_node("write_introduction", write_introduction)
    builder.add_node("write_conclusion", write_conclusion)
    builder.add_node("finalize_report", finalize_report)
    
    builder.add_edge(START, "setting_consultants")
    builder.add_edge("setting_consultants", "initiate_all_interviews")
    builder.add_conditional_edges(
        "initiate_all_interviews",
        route_to_interviews,
        ["conduct_interview"]
    )
    
    builder.add_edge("conduct_interview", "write_introduction")
    builder.add_edge("write_introduction", "write_report")
    builder.add_edge("write_report", "write_conclusion")
    builder.add_edge("write_conclusion", "finalize_report")
    builder.add_edge("finalize_report", END)

    memory = MemorySaver()
    
    return builder.compile(checkpointer=memory)

def invoke_graph(
    graph: CompiledStateGraph,
    inputs: dict,
    config: RunnableConfig,
    node_names: List[str] = [],
    callback: Callable = None,
):
    """LangGraph 실행"""
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
                    print(f"Node: {node_name}")
                else:
                    print(f"Node: {node_name} in [{formatted_namespace}]")
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
                            print(f"{k}:\n{v}")
                print("=" * 50)

def save_report_to_db(report: str, report_type: str = "daily"):
    """보고서 DB 저장"""
    DB_URI = "postgresql+psycopg2://postgres:1234@localhost:5433/kalantir"
    
    db = SQLDatabase.from_uri(
        DB_URI,
        engine_args={
            "connect_args": {
                "options": "-c client_encoding=UTF8 -c search_path=ai"
            }
        }
    )
    
    file_path = make_pdf_and_save(report)
    
    query = text("""
    INSERT INTO reports (file_path, contents, report_type)
    VALUES (:file_path, :final_report, :report_type)
    RETURNING id;
    """)
    
    params = {"file_path": file_path, "final_report": report, "report_type": report_type}
    
    with db._engine.connect() as conn:
        result = conn.execute(query, params)
        new_id = result.scalar()
        conn.commit()
    
    return new_id

def main():
    """메인 실행"""
    graph = build_research_graph()
    
    config = {"configurable": {"thread_id": "1"}}
    
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
    
    print("=" * 80)
    print("Starting Research Process...")
    print("=" * 80)
    
    invoke_graph(graph, initial_inputs, config)
    
    final_state = graph.get_state(config)
    final_report = final_state.values.get("final_report")
    
    print("\n" + "=" * 80)
    print("FINAL REPORT")
    print("=" * 80)
    print(final_report)
    
    report_id = save_report_to_db(final_report, "daily")
    print(f"\nReport saved to database with ID: {report_id}")
    
    return final_report

if __name__ == "__main__":
    main()