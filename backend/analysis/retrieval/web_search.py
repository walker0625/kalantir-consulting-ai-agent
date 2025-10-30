"""웹 검색"""
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.prompts import load_prompt
from langchain_tavily import TavilySearch
from util.path import PROMPT_DIR

from backend.analysis.nodes.states import InterviewState, SearchQuery

load_dotenv()

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.7)

def search_web(state: InterviewState):
    """웹 검색으로 문서를 찾는 노드"""
    consultant = state['consultant']
    topic = state['topic']
    
    search_instructions = load_prompt(PROMPT_DIR / "search_instructions.yaml", encoding="utf-8").format(
        name=consultant.name, role=consultant.role, 
        department=consultant.department, description=consultant.description, 
        topic=topic, domain_keywords=consultant.domain_keywords, 
        search_focus=consultant.search_focus)
    
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([SystemMessage(search_instructions)] + state['messages'])
    
    tavily_search = TavilySearch(max_results=6)
    search_docs = tavily_search.invoke(search_query.search_query)
    
    return {"context": [search_docs]}