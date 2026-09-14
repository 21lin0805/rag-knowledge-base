from dotenv import load_dotenv
import os

# 加载 .env 里的 API Key
load_dotenv()

from app.rag.loader import load_document
from app.rag.pipeline import RAGPipeline

print("开始加载文档...")
docs = load_document('data/documents/test.txt')

print("开始把文档存入向量库...")
pipeline = RAGPipeline()
count = pipeline.ingest(docs)
print(f"成功入库 {count} 个文本块")

print("开始提问...")
result = pipeline.query('什么是BM25')
print(f"回答: {result['answer']}")
print(f"耗时: {result['latency_ms']}ms")