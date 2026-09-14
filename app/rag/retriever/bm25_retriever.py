import jieba
from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, documents):
        self.documents = documents
        # 用 jieba 对每个文档块分词
        self.tokenized_corpus = [
            list(jieba.cut(doc.page_content)) for doc in documents
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query, top_k=5):
        # 对查询分词
        tokenized_query = list(jieba.cut(query))
        # 计算每个文档的 BM25 分数
        scores = self.bm25.get_scores(tokenized_query)
        # 取分数最高的 top_k 个
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]
        return [self.documents[i] for i in top_indices]