"""State 정의"""
import operator
from typing import Annotated, List
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState

from util.persona import Consultant

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