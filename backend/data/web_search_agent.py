import os
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from dotenv import load_dotenv

from langchain.prompts import load_prompt
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Qdrant
from langchain_mcp_adapters.client import MultiServerMCPClient

from util.path import PROMPT_DIR

load_dotenv()

class WebSearchMCPAgent:
    """MCP 기반 웹 검색 Agent (Exa, Brave, Perplexity)"""
    
    def __init__(
        self,
        qdrant_url: str,
        collection_name: str = "industry_ai_cases",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        embedding_model: str = "text-embedding-3-small"
    ):
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding = OpenAIEmbeddings(model=embedding_model)
        
        self.SMITHERY_API_KEY = os.getenv('SMITHERY_API_KEY')
        self.SMITHERY_API_PROFILE = os.getenv('SMITHERY_API_PROFILE')
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.mcp_tools = None
        
        # 프롬프트 파일 매핑
        self.prompt_configs = {
            'exa': {
                'display_name': 'EXA',
                'queries': [
                    {'file': 'mcp_exa_technical.yaml', 'name': 'technical'},
                    {'file': 'mcp_exa_research.yaml', 'name': 'research'}
                ]
            },
            'brave': {
                'display_name': 'BRAVE',
                'queries': [
                    {'file': 'mcp_brave_korean.yaml', 'name': 'korean_news'},
                    {'file': 'mcp_brave_global.yaml', 'name': 'global_news'}
                ]
            },
            'perplexity': {
                'display_name': 'PERPLEXITY',
                'queries': [
                    {'file': 'mcp_perplexity_comprehensive.yaml', 'name': 'comprehensive'},
                    {'file': 'mcp_perplexity_comparison.yaml', 'name': 'comparison'}
                ]
            }
        }
        
    async def create_mcp_client(self):
        """MCP 클라이언트 생성"""
        print("🔧 MCP 클라이언트 연결 중...")
        
        try:
            client = MultiServerMCPClient({
                "exa_search_mcp": {
                    "transport": "stdio",
                    "command": "npx",
                    "args": ["-y", "@smithery/cli@latest", "run", "exa",
                            "--key", self.SMITHERY_API_KEY,
                            "--profile", self.SMITHERY_API_PROFILE]
                },
                "brave_search_mcp": {
                    "transport": "stdio",
                    "command": "npx",
                    "args": ["-y", "@smithery/cli@latest", "run", "@JonyanDunh/brave-search-mcp",
                            "--key", self.SMITHERY_API_KEY,
                            "--profile", self.SMITHERY_API_PROFILE]
                },
                "perplexity_search": {
                    "transport": "stdio",
                    "command": "npx",
                    "args": ["-y", "@smithery/cli@latest", "run", "@arjunkmrm/perplexity-search",
                            "--key", self.SMITHERY_API_KEY,
                            "--profile", self.SMITHERY_API_PROFILE]
                }
            })
            
            self.mcp_tools = await asyncio.wait_for(
                client.get_tools(),
                timeout=30
            )
            
            print(f"✅ MCP 연결 완료! 도구: {len(self.mcp_tools)}개")
            
        except asyncio.TimeoutError:
            print("❌ MCP 도구 로드 타임아웃 (30초)")
            raise Exception("MCP client connection timeout")
        except Exception as e:
            print(f"❌ MCP 클라이언트 연결 실패: {str(e)}")
            raise

    async def search_with_mcp(self, query: str, engine_name: str, timeout: int = 180) -> Dict:
        """MCP 도구로 검색 실행"""
        print(f"🔍 [{engine_name.upper()}] 검색 중... (타임아웃: {timeout}초)")
        
        if not self.mcp_tools:
            return {
                "engine": engine_name,
                "content": "",
                "error": "MCP tools not loaded",
                "success": False
            }
        
        try:
            start_time = datetime.now()
            
            agent_executor = create_react_agent(
                model=self.llm,
                tools=self.mcp_tools
            )
            
            try:
                response = await asyncio.wait_for(
                    agent_executor.ainvoke({
                        "messages": [HumanMessage(content=query)]
                    }),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                duration = (datetime.now() - start_time).total_seconds()
                print(f"⏱️  타임아웃! ({duration:.1f}초)")
                return {
                    "engine": engine_name,
                    "content": "",
                    "error": f"Timeout after {timeout} seconds",
                    "duration": duration,
                    "timestamp": datetime.now(),
                    "success": False
                }
            
            duration = (datetime.now() - start_time).total_seconds()
            output = response["messages"][-1].content
            
            print(f"✅ 완료: {len(output):,}자 ({duration:.1f}초)")
            
            return {
                "engine": engine_name,
                "content": output,
                "length": len(output),
                "duration": duration,
                "timestamp": datetime.now(),
                "success": True
            }
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            print(f"❌ 검색 실패: {str(e)}")
            
            return {
                "engine": engine_name,
                "content": "",
                "error": str(e),
                "duration": duration,
                "timestamp": datetime.now(),
                "success": False
            }

    async def search_all_engines(self) -> List[Dict]:
        """모든 엔진으로 검색"""
        print("🚀 전체 검색 엔진 실행 시작")
        
        all_results = []
        total_queries = sum(len(config['queries']) for config in self.prompt_configs.values())
        current_query = 0
        
        for engine_key, prompt_config in self.prompt_configs.items():
            display_name = prompt_config['display_name']
            queries = prompt_config['queries']
            
            print(f"\n📚 [{display_name}] ({len(queries)}개 쿼리)")
            
            for query_info in queries:
                current_query += 1
                yaml_file = query_info['file']
                query_name = query_info['name']
                
                print(f"[{current_query}/{total_queries}] {query_name}")
                
                try:
                    prompt_template = load_prompt(PROMPT_DIR / yaml_file, encoding="utf-8")
                    query_text = prompt_template.template
                    
                    # 타임아웃 동적 조정
                    if len(query_text) > 2000:
                        timeout = 240
                    elif len(query_text) > 1000:
                        timeout = 180
                    else:
                        timeout = 120
                    
                    engine_name = f"{engine_key}_{query_name}"
                    
                    result = await self.search_with_mcp(
                        query=query_text,
                        engine_name=engine_name,
                        timeout=timeout
                    )
                    
                    if result['success']:
                        all_results.append(result)
                    else:
                        print(f"⚠️  실패: {result.get('error', 'Unknown')}")
                
                except Exception as e:
                    print(f"❌ 오류: {str(e)}")
        
        print(f"\n✅ 검색 완료: {len(all_results)}/{total_queries}개 성공")
        
        return all_results

    def synthesize_results(self, results: List[Dict]) -> str:
        """검색 결과 종합"""
        print("📊 검색 결과 종합 중...")
        
        if not results:
            print("⚠️  검색 결과 없음")
            return "검색 결과 없음"
        
        collected_data = "\n\n" + "="*70 + "\n\n"
        
        for i, result in enumerate(results, 1):
            header = f"""
[{i}/{len(results)}] {result['engine'].upper()}
수집 시간: {result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
데이터 크기: {result['length']:,}자
소요 시간: {result['duration']:.1f}초
{'-'*70}
"""
            collected_data += header + result['content'] + "\n\n" + "="*70 + "\n\n"
        
        search_summary_template = load_prompt(
            PROMPT_DIR / "mcp_web_search_summary.yaml", 
            encoding="utf-8"
        )
        
        search_summary = search_summary_template.format(collected_data=collected_data)
        
        print("🤖 LLM 종합 분석 중...")
        response = self.llm.invoke([HumanMessage(content=search_summary)])
        final_report = response.content
        
        complete_content = collected_data + "\n\n" + "="*70 + "\n" + \
                        "[📋 최종 종합 분석 리포트]" + "\n" + "="*70 + "\n\n" + \
                        final_report
        
        print(f"✅ 종합 완료: {len(complete_content):,}자")
        
        return complete_content

    def split_text(self, text: str) -> List[Document]:
        """텍스트 청크 분할"""
        print("📄 텍스트 분할 중...")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]
        )
        
        chunks = splitter.split_text(text)
        today = datetime.now().strftime("%Y%m%d")
        
        docs = [
            Document(
                page_content=chunk,
                metadata={
                    "source": f"{self.collection_name}_{today}",
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "collection_date": today,
                    "topic": "ai_adoption_cases"
                }
            )
            for i, chunk in enumerate(chunks)
        ]
        
        print(f"✅ 분할 완료: {len(docs)}개")
        return docs

    def store_in_qdrant(self, docs: List[Document]):
        """Qdrant에 저장"""
        print(f"💾 Qdrant 저장 중... (Collection: {self.collection_name})")
        
        try:
            Qdrant.from_documents(
                documents=docs,
                embedding=self.embedding,
                collection_name=self.collection_name,
                url=self.qdrant_url
            )
            print(f"✅ 저장 완료: {len(docs)}개 문서")
        except Exception as e:
            print(f"❌ 저장 실패: {str(e)}")
            raise

    async def run_async(self) -> Dict:
        """비동기 실행"""
        start_time = datetime.now()
        
        print("\n" + "=" * 70)
        print("🎯 웹 검색 MCP Agent 시작")
        print("=" * 70)
        
        try:
            # 1. MCP 클라이언트 생성
            await self.create_mcp_client()
            
            # 2. 검색 실행
            results = await self.search_all_engines()
            
            # 3. 결과 종합
            complete_content = self.synthesize_results(results)
            
            # 4. 텍스트 분할
            docs = self.split_text(complete_content)
            
            # 5. Qdrant 저장
            self.store_in_qdrant(docs)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            print("\n" + "=" * 70)
            print("🎉 웹 검색 완료!")
            print(f"📊 검색: {len(results)}회")
            print(f"📦 저장: {len(docs)}개 문서")
            print(f"⏱️  소요: {duration:.1f}초 ({duration/60:.1f}분)")
            print("=" * 70 + "\n")
            
            return {
                "success": True,
                "search_count": len(results),
                "docs_count": len(docs),
                "duration": duration
            }
            
        except Exception as e:
            print(f"\n❌ 오류: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    def run(self) -> Dict:
        """동기 래퍼 (Supervisor에서 호출용)"""
        return asyncio.run(self.run_async())