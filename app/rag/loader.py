from langchain_community.document_loaders import TextLoader, PyPDFLoader
from pathlib import Path

LOADER_MAP = {
    ".txt": TextLoader,
    ".pdf": PyPDFLoader,
}

def load_document(file_path: str):
    """加载文档，返回 LangChain Document 对象列表"""
    ext = Path(file_path).suffix.lower()
    loader_cls = LOADER_MAP.get(ext)
    if not loader_cls:
        raise ValueError(f"不支持的格式: {ext}，目前支持 .txt 和 .pdf")
    
    if ext == ".txt":
        loader = loader_cls(file_path, encoding="utf-8")
    else:
        loader = loader_cls(file_path)
    return loader.load()