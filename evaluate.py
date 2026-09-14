import time
from dotenv import load_dotenv
load_dotenv()

from app.rag.loader import load_document
from app.rag.splitter import split_documents
from app.db.vector_store import get_vector_store, clear_vector_store, add_documents
from app.rag.retriever.hybrid_retriever import hybrid_search
from app.rag.reranker import rerank
from eval_dataset import EVAL_DATASET


def is_hit(doc, keywords):
    """判断文档块是否包含任意一个关键词"""
    content = doc.page_content
    return any(kw in content for kw in keywords)


def evaluate_method(name, search_fn, top_k=3):
    """通用评估：对数据集跑一遍，计算 Recall@K 和 MRR"""
    recall_hits = 0
    mrr_sum = 0.0

    print(f"\n{'=' * 60}")
    print(f"评估方法：{name}")
    print(f"{'=' * 60}")

    for item in EVAL_DATASET:
        q = item["question"]
        keywords = item["keywords"]

        # 检索
        results = search_fn(q, top_k)

        # 计算 Recall@K（只要前 K 个里有命中就算 1）
        hit = any(is_hit(doc, keywords) for doc in results)
        recall_hits += int(hit)

        # 计算 MRR（第一个命中的排名的倒数）
        for rank, doc in enumerate(results, 1):
            if is_hit(doc, keywords):
                mrr_sum += 1.0 / rank
                break

        status = "✅" if hit else "❌"
        print(f"  {status}  {q}")

    n = len(EVAL_DATASET)
    recall = recall_hits / n
    mrr = mrr_sum / n

    print(f"\n  Recall@{top_k}: {recall:.4f}")
    print(f"  MRR:         {mrr:.4f}")
    return {"Recall": recall, "MRR": mrr}


def main():
    # 1. 准备知识库
    clear_vector_store()
    docs = load_document("data/documents/test.txt")
    chunks = split_documents(docs, chunk_size=300, chunk_overlap=50)
    add_documents(chunks)
    print(f"入库 {len(chunks)} 个文本块")

    # 2. 加载全部文档用于 BM25（BM25 需要完整列表）
    all_chunks = chunks

    # 3. 定义三种检索方法
    vs = get_vector_store()

    def vector_only(query, k):
        return vs.similarity_search(query, k=k)

    def hybrid_only(query, k):
        return hybrid_search(query, all_chunks, top_k=k)

    def hybrid_with_rerank(query, k):
        candidates = hybrid_search(query, all_chunks, top_k=10)
        return rerank(query, candidates, top_k=k)

    # 4. 依次评估
    results = {}
    results["纯向量检索"] = evaluate_method("纯向量检索", vector_only)
    results["混合检索"] = evaluate_method("混合检索（向量+BM25）", hybrid_only)
    results["混合检索+Rerank"] = evaluate_method(
        "混合检索 + Rerank", hybrid_with_rerank
    )

    # 5. 汇总
    print(f"\n{'=' * 60}")
    print("汇总对比")
    print(f"{'=' * 60}")
    print(f"{'方法':<25}{'Recall@3':<12}{'MRR':<10}")
    for name, m in results.items():
        print(f"{name:<25}{m['Recall']:<12.4f}{m['MRR']:<10.4f}")


if __name__ == "__main__":
    main()