from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents, chunk_size=300, chunk_overlap=50):
    """将文档切分成小块，保留语义边界"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        length_function=len,
    )
    return splitter.split_documents(documents)