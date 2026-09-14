import os
from modelscope import snapshot_download
from langchain_community.embeddings import HuggingFaceEmbeddings

_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        model_dir = snapshot_download("AI-ModelScope/bge-small-zh-v1.5")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        _embedding_model = HuggingFaceEmbeddings(
            model_name=model_dir,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    return _embedding_model