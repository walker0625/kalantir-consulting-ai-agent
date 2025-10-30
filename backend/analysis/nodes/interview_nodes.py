"""인터뷰 노드들"""
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, get_buffer_string
from langchain_core.prompts import load_prompt

from util.path import PROMPT_DIR

from backend.analysis.nodes.states import InterviewState

load_dotenv()

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.7)

def ask_question(state: InterviewState):
    """컨설턴트가 질문을 생성하는 노드"""
    topic = state['topic']
    consultant = state['consultant']
    messages = state['messages']
    
    question_instructions = load_prompt(PROMPT_DIR / "question_instructions.yaml", encoding="utf-8")
    
    system_message = question_instructions.format(topic=topic, goals=consultant.persona)
    question = llm.invoke([SystemMessage(content=system_message)] + messages)
    
    return {'messages': [question]}

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
    context = state['context'] + [interview] # 페르소나와의 인터뷰 내용도 보고서 작성에 활용
    consultant = state['consultant']
    
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