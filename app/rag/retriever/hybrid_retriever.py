from app.db.vector_store import get_vector_store
from app.rag.retriever.bm25_retriever import BM25Retriever


def reciprocal_rank_fusion(results_list, k=60):
    """
    RRF 融合：把多路检索的结果按排名融合成一个统一排序。
    k 是平滑参数，经验值 60。
    """
    fused_scores = {}
    for results in results_list:
        for rank, doc in enumerate(results):
            # 用内容前 100 字作简易 ID（真实项目应使用文档 ID）
            doc_id = doc.page_content[:100]
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {"doc": doc, "score": 0.0}
            # RRF 公式：1 / (k + rank)
            fused_scores[doc_id]["score"] += 1.0 / (k + rank + 1)

    sorted_items = sorted(
        fused_scores.values(), key=lambda x: x["score"], reverse=True
    )
    return [item["doc"] for item in sorted_items]


def hybrid_search(query, documents, top_k=10):
    """混合检索：向量检索 + BM25，用 RRF 融合"""
    # 1. 向量检索
    vs = get_vector_store()
    vector_results = vs.similarity_search(query, k=top_k)

    # 2. BM25 检索
    bm25 = BM25Retriever(documents)
    bm25_results = bm25.search(query, top_k=top_k)

    # 3. RRF 融合
    fused = reciprocal_rank_fusion([vector_results, bm25_results])
    return fused[:top_k]