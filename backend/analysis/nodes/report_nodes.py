"""보고서 작성 노드들"""
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import load_prompt

from util.path import PROMPT_DIR
from util.persona import Consultant, CONSULTANT_PROFILES

from backend.analysis.nodes.states import ResearchGraphState

load_dotenv()

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.7)

def setting_consultants(state: ResearchGraphState):
    """컨설턴트 설정"""
    max_consultants = state.get("max_consultants", 3)
    
    consultants = [
        Consultant(**profile) 
        for profile in CONSULTANT_PROFILES[:max_consultants]
    ]
    
    return {'consultants': consultants}

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
    
    introduction_instructions = load_prompt(PROMPT_DIR / "introduction_instructions.yaml", encoding="utf-8")
    
    instructions = introduction_instructions.format(
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
    
    conclusion_instructions = load_prompt(PROMPT_DIR / "conclusion_instructions.yaml", encoding="utf-8")
    
    instructions = conclusion_instructions.format(
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