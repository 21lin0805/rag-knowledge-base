import os
from langchain_chroma import Chroma
from app.rag.embedder import get_embedding_model

def get_vector_store(collection_name="knowledge_base"):
    """获取 Chroma 向量库实例（持久化到本地磁盘）"""
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embedding_model(),
        persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db"),
    )

def add_documents(chunks):
    """将文本块向量化后存入 Chroma"""
    vs = get_vector_store()
    vs.add_documents(chunks)
    return len(chunks)

def clear_vector_store(collection_name="knowledge_base"):
    """清空整个向量库集合"""
    vs = get_vector_store(collection_name=collection_name)
    try:
        vs.delete_collection()
        print(f"[清空] 已删除集合 {collection_name}")
    except Exception as e:
        print(f"[清空] 集合不存在或删除失败: {e}")