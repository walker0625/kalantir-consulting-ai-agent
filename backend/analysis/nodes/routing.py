"""라우팅 함수들"""
from langchain_core.messages import HumanMessage
from langgraph.constants import Send

from backend.analysis.nodes.states import ResearchGraphState

def initiate_all_interviews_node(state: ResearchGraphState):
    """병렬 실행용 더미 노드"""
    return {}

def route_to_interviews(state: ResearchGraphState):
    """인터뷰 라우팅 함수"""
    max_consultants = state.get('max_consultants', 3)
    
    return [
        Send("conduct_interview", 
                {
                "consultant": consultant,
                "messages": [
                    HumanMessage(content="Welcome! We'll begin the interview now.")
                ],
                "max_num_turns": 3,
                "context": [],
                "topic": state['topic'],
                "sections": state['sections']
            }
        )
        for consultant in state['consultants'][:max_consultants]
    ]