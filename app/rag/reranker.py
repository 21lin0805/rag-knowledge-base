import os
from modelscope import snapshot_download
from sentence_transformers import CrossEncoder

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        # 从魔搭下载/加载模型
        model_dir = snapshot_download("Xorbits/bge-reranker-base")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        _reranker = CrossEncoder(model_dir, max_length=512)
    return _reranker


def rerank(query, documents, top_k=3):
    """对候选文档做重排序，返回最相关的 top_k 个"""
    if not documents:
        return []

    reranker = get_reranker()
    # 构造 (query, doc) 对
    pairs = [[query, doc.page_content] for doc in documents]
    # 计算相关性分数
    scores = reranker.predict(pairs)
    # 按分数降序排列
    scored_docs = sorted(
        zip(documents, scores), key=lambda x: x[1], reverse=True
    )
    return [doc for doc, _ in scored_docs[:top_k]]