import time
from app.rag.splitter import split_documents
from app.db.vector_store import add_documents, clear_vector_store
from app.rag.retriever.hybrid_retriever import hybrid_search
from app.rag.reranker import rerank
from app.rag.generator import generate_answer


class RAGPipeline:
    def __init__(self):
        self.documents = []

    def ingest(self, raw_documents, chunk_size=300, chunk_overlap=50, reset=True):
        """
        文档入库：切分 -> 向量化 -> 存入 Chroma
        reset=True 时，先清空旧向量库，避免重复入库
        """
        if reset:
            clear_vector_store()
            self.documents = []

        chunks = split_documents(raw_documents, chunk_size, chunk_overlap)
        self.documents.extend(chunks)
        count = add_documents(chunks)
        return count

    def query(self, question, top_k_retrieve=10, top_k_rerank=3):
        """问答：混合检索（粗召回） -> Rerank（精排） -> 生成"""
        start = time.time()

        # 1. 混合检索：召回更多候选
        candidates = hybrid_search(
            question, self.documents, top_k=top_k_retrieve
        )

        # 2. Rerank 精排
        final_docs = rerank(question, candidates, top_k=top_k_rerank)

        # 3. 调试打印
        print("=" * 50)
        print(f"混合检索召回 {len(candidates)} 个，Rerank 后保留 {len(final_docs)} 个：")
        for i, doc in enumerate(final_docs):
            print(f"  [{i}] {doc.page_content[:80]}...")
        print("=" * 50)

        # 4. 生成回答
        result = generate_answer(question, final_docs)
        result["latency_ms"] = round((time.time() - start) * 1000, 2)
        result["retrieved_count"] = len(candidates)
        result["final_count"] = len(final_docs)
        return result