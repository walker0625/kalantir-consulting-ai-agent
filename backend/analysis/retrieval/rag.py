"""RAG 검색 및 Reranking"""
from datetime import datetime
import math
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_community.document_transformers import LongContextReorder
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from backend.analysis.nodes.states import InterviewState

def create_retriever(collection_name: str, k: int = 40):
    """MMR 기반 retriever 생성"""
    client = QdrantClient(url='http://localhost:6333')
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vectorstore = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings
    )
    
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": k * 3,
            "lambda_mult": 0.7
        }
    )

def calculate_time_weight(processed_at: str) -> float:
    """시간 가중치 계산"""
    try:
        doc_date = datetime.fromisoformat(processed_at)
        days_old = (datetime.now() - doc_date).days
        return math.exp(-0.1 * days_old / 365)
    except:
        return 0.5

def bm25_search(query: str, docs: list) -> dict:
    """BM25 키워드 검색"""
    if not docs:
        return {}
    
    tokenized_docs = [doc.page_content.lower().split() for doc in docs]
    tokenized_query = query.lower().split()
    
    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(tokenized_query)
    
    max_score = max(scores) if max(scores) > 0 else 1
    
    return {id(doc): score / max_score for doc, score in zip(docs, scores)}

def hybrid_rerank(query: str, vector_docs: list, top_k: int = 30) -> list:
    """Hybrid Reranking: Vector(40%) + BM25(30%) + Time(30%)"""
    if not vector_docs:
        return []
    
    bm25_scores = bm25_search(query, vector_docs)
    
    doc_scores = []
    for idx, doc in enumerate(vector_docs):
        doc_id = id(doc)
        
        vector_score = 1.0 - (idx / len(vector_docs))
        bm25_score = bm25_scores.get(doc_id, 0)
        time_weight = calculate_time_weight(
            doc.metadata.get('processed_at', '')
        ) if hasattr(doc, 'metadata') else 0.5
        
        final_score = (
            vector_score * 0.4 +
            bm25_score * 0.3 +
            time_weight * 0.3
        )
        
        doc_scores.append((final_score, doc))
    
    doc_scores.sort(key=lambda x: x[0], reverse=True)
    
    return [doc for _, doc in doc_scores[:top_k]]

def remove_duplicates(docs: list) -> list:
    """문서 중복 제거"""
    seen = set()
    unique_docs = []
    
    for doc in docs:
        file_name = doc.metadata.get('file_name', '') if hasattr(doc, 'metadata') else ''
        page = doc.metadata.get('page', 0) if hasattr(doc, 'metadata') else 0
        
        key = f"{file_name}_{page}"
        
        if key not in seen or not file_name:
            seen.add(key)
            unique_docs.append(doc)
    
    return unique_docs


# RAG 검색 파이프라인 (5단계)
# 1. MMR Vector Search (다양성) + 중복 제거 로직(remove_duplicates())
# 2. BM25 Keyword Search (정확도)
# 3. Time Weighting (최신성)
# 4. Hybrid Reranking (통합 점수)
# 5. LITM Reordering (최종 재정렬)
def search_rag(state: InterviewState):
    """일일보고서용 RAG 검색 파이프라인"""
    COLLECTIONS = ['industry_ai_cases', 'samsung_internal_db', 'samsung_external_pdf']
    
    query = state['topic']
    all_results = []
    
    print("\n[RAG 검색 파이프라인 시작]")
    for collection_name in COLLECTIONS:
        retriever = create_retriever(collection_name, k=40)
        results = retriever.invoke(query)
        all_results.extend(results)
        print(f"[MMR] {collection_name}: {len(results)}개 문서")
    
    print(f"\n[초기 검색] {len(all_results)}개 문서")
    
    all_results = remove_duplicates(all_results)
    print(f"[중복 제거] {len(all_results)}개 문서")
    
    reranked_results = hybrid_rerank(query, all_results, top_k=30)
    print(f"[Hybrid Rerank] {len(reranked_results)}개 문서 선택")
    
    final_results = reranked_results[:7]
    final_results = LongContextReorder().transform_documents(final_results)
    print(f"[LITM Reorder] {len(final_results)}개 최종 선택\n")
    
    print("[최종 선택 문서]")
    for i, doc in enumerate(final_results, 1):
        file_name = doc.metadata.get('file_name', 'unknown')
        doc_date = doc.metadata.get('processed_at', 'N/A')[:10] if 'processed_at' in doc.metadata else 'N/A'
        preview = doc.page_content[:80].replace('\n', ' ')
        print(f"  {i}. {file_name} ({doc_date})")
        print(f"     \"{preview}...\"")
    print("-" * 60 + "\n")
    
    return {"context": [final_results]}