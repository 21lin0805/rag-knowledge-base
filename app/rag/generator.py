import os
from langchain_openai import ChatOpenAI

PROMPT_TEMPLATE = """你是一个知识库问答助手。请根据以下参考资料回答问题。
如果参考资料中没有相关信息，请直接说"根据现有资料无法回答该问题"，不要编造。

参考资料：
{context}

问题：{question}

回答要求：
1. 优先使用参考资料中的信息回答
2. 只要参考资料包含与问题相关的内容，就直接回答，不要过度谨慎
3. 只有当参考资料与问题完全无关时，才回答"根据现有资料无法回答该问题"
4. 回答要简洁、直接

回答："""

def get_llm():
    return ChatOpenAI(
        model="glm-4-flash",                        # 智谱免费模型
        api_key=os.getenv("ZHIPU_API_KEY"),
        base_url="https://open.bigmodel.cn/api/paas/v4/",  # 智谱的 OpenAI 兼容地址
        temperature=0.1,
    )

def generate_answer(question, context_docs):
    context = "\n\n---\n\n".join([doc.page_content for doc in context_docs])
    llm = get_llm()
    response = llm.invoke(PROMPT_TEMPLATE.format(
        context=context, question=question
    ))
    return {
        "answer": response.content,
        "sources": [
            {"content": doc.page_content[:200], "metadata": doc.metadata}
            for doc in context_docs
        ],
    }