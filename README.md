# 私有文档 RAG 知识库问答系统

基于 LangChain 构建的私有文档检索增强生成（RAG）系统，支持 txt/pdf 导入、混合检索、重排序、量化评估与 REST API 服务。

## 技术栈

- **框架**：Python 3.10、LangChain
- **大模型**：GLM-4-Flash（智谱 AI，OpenAI 兼容协议）
- **向量模型**：BGE-small-zh-v1.5（本地部署）
- **重排序**：BGE-Reranker-Base（Cross-Encoder）
- **向量库**：Chroma
- **检索**：BM25 + 向量混合检索（RRF 融合）
- **工程化**：FastAPI、MySQL

## 系统架构
文档 → 分块 → 向量化 → Chroma
↓
用户提问 → 混合检索（向量 + BM25）→ RRF 融合 → Rerank → 大模型生成 → 回答
↓
MySQL 日志

## 核心特性

- 支持 txt/pdf 文档导入与智能分块
- BM25 + 向量混合检索，RRF 融合两路结果
- BGE-Reranker 重排序，提升检索精度
- 防幻觉 Prompt 设计，约束模型基于上下文回答
- 量化评估（Recall@K、MRR）
- FastAPI 提供 REST 接口
- MySQL 持久化问答日志并统计耗时

## 评估结果

基于 12 条测试集，对比三种检索方案：

| 方法 | Recall@3 | MRR |
|------|----------|-----|
| 纯向量检索 | 0.9167 | 0.8750 |
| 混合检索（向量+BM25） | 1.0000 | 0.9444 |
| **混合检索 + Rerank** | **1.0000** | **1.0000** |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt