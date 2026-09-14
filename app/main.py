import os
import json
import shutil
import pymysql
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from app.rag.loader import load_document
from app.rag.pipeline import RAGPipeline

load_dotenv()

app = FastAPI(title="RAG 知识库问答 API", version="1.0")
pipeline = RAGPipeline()


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


def log_to_mysql(question: str, result: dict):
    """将问答日志写入 MySQL"""
    try:
        conn = pymysql.connect(
            host=os.getenv("MYSQL_HOST", "localhost"),
            port=int(os.getenv("MYSQL_PORT", 3306)),
            user=os.getenv("MYSQL_USER", "root"),
            password=os.getenv("MYSQL_PASSWORD", ""),
            database=os.getenv("MYSQL_DATABASE", "rag_logs"),
            charset="utf8mb4",
        )
        with conn.cursor() as cursor:
            cursor.execute(
                """INSERT INTO qa_logs
                   (question, answer, sources, latency_ms, retrieved_count)
                   VALUES (%s, %s, %s, %s, %s)""",
                (
                    question,
                    result["answer"],
                    json.dumps(result["sources"], ensure_ascii=False),
                    result.get("latency_ms", 0),
                    result.get("retrieved_count", 0),
                ),
            )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[MySQL] 写入失败：{e}")


@app.get("/")
def root():
    return {"message": "RAG API 已启动，请访问 /docs 查看接口文档"}


@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """上传文档并入库"""
    save_path = f"./data/documents/{file.filename}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    docs = load_document(save_path)
    count = pipeline.ingest(docs, reset=True)
    return {"message": f"成功导入 {count} 个文本块", "file": file.filename}


@app.post("/query")
async def query(req: QueryRequest):
    """问答查询"""
    result = pipeline.query(req.question, top_k_rerank=req.top_k)
    log_to_mysql(req.question, result)
    return result