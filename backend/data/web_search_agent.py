from dotenv import load_dotenv
from datetime import datetime
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_tavily import TavilySearch
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Qdrant

load_dotenv()

class WebSearchAgent:
    """웹 검색 결과를 임베딩 후 Qdrant에 저장하는 클래스"""
    def __init__(
        self,
        query: str,
        chunk_size: int = 300,
        chunk_overlap: int = 30,
        embedding_model: str = "text-embedding-3-small",
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "samsung_external_web"
    ):
        self.query = query
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding = OpenAIEmbeddings(model=embedding_model)
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name

        self.llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
        self.tavily_search = TavilySearch(max_result=3)
        self.embedding = OpenAIEmbeddings(model="text-embedding-3-small")
        
    # TODO 
    # 1. Tavily를 MCP 웹서치(3종-Exa/Brave/Perplexity)로 변경
    # 2. prompt를 좀 더 구체적이고 사업에 관련된 정보를 받아올 수 있도록 수정
    def create_agent(self):
        """LLM + Tavily 도구로 AgentExecutor 생성"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "너는 툴만 사용해서 사실을 확인하고 간단하게 대답해줘"),
            ("human", self.query),
            ("placeholder", "{agent_scratchpad}")
        ])

        agent = create_openai_tools_agent(
            llm=self.llm,
            tools=[self.tavily_search],
            prompt=prompt
        )
        
        return AgentExecutor(agent=agent, tools=[self.tavily_search], verbose=True)

    def search_web(self, executor: AgentExecutor) -> str:
        """웹 검색 및 결과 요약"""
        print(f"Searching web for query: {self.query}")
        result = executor.invoke({"input": self.query})
        print("Web search completed.\n")
        
        return result["output"]

    def split_text(self, text: str):
        """텍스트를 문서 리스트로 분할"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        chunks = splitter.split_text(text)
        today = datetime.now().strftime("%Y%m%d")

        docs = [
            Document(
                page_content=chunk,
                metadata={"source": f"{self.collection_name}_{today}", "chunk_id": i}
            )
            for i, chunk in enumerate(chunks)
        ]
        print(f"Split into {len(docs)} chunks.")
        
        return docs

    def store_in_qdrant(self, docs):
        """Qdrant에 문서 저장"""
        print(f"Saving to Qdrant collection: {self.collection_name}")
        Qdrant.from_documents(
            documents=docs,
            embedding=self.embedding,
            collection_name=self.collection_name,
            url=self.qdrant_url
        )
        print(f"Successfully stored {len(docs)} docs in '{self.collection_name}'.\n")

    def run(self):
        executor = self.create_agent()
        output_text = self.search_web(executor)
        docs = self.split_text(output_text)
        self.store_in_qdrant(docs)

if __name__ == "__main__":
    webSearchAgent = WebSearchAgent(
        query="삼성전자와 ai 관련된 내용을 모두 찾아줘",
        chunk_size=300,
        chunk_overlap=30,
        embedding_model="text-embedding-3-small",
        qdrant_url="http://localhost:6333",
        collection_name="samsung_external_web"
    )
    webSearchAgent.run()
