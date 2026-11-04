# RAG 智能问答系统（RAG + Agent + 评测 + MCP）

基于 DeepSeek + ChromaDB 的本地知识库问答系统。从零手写 RAG 检索链路、Agent 工具调用，并搭建了完整的评测体系。

## 技术栈

- **大模型**：DeepSeek（OpenAI 兼容接口）
- **向量库**：ChromaDB（本地持久化）
- **Embedding**：`paraphrase-multilingual-MiniLM-L12-v2`
- **重排**：`bge-reranker-base`（CrossEncoder）
- **检索**：向量 + BM25 混合召回 → RRF 融合 → 重排精排
- **框架**：手写版 + LangChain / LangGraph 版 + FastAPI Web + MCP

## 文件结构

| 文件 | 作用 |
|------|------|
| `rag_system.py` | RAG 核心：切分、混合检索、重排、多轮问答 |
| `eval_rag.py` | 四指标评测（召回 / 正确 / 反幻觉 / 忠实 + LLM-as-Judge） |
| `compare_rerank.py` | 重排消融实验（无重排 50% → 有重排 100%） |
| `debug_badcase.py` | badcase 分层排查（评测/检索/生成/裁判四层定位） |
| `agent.py` | 手写 ReAct Agent（function calling + 多步循环） |
| `agent_graph.py` | LangGraph 版 Agent |
| `langchain_rag.py` | LangChain 版 RAG |
| `web_ui.py` | FastAPI 网页问答 |
| `import_my_docs.py` | 知识库数据导入 |
| `mcp/` | MCP server（RAG + weather） |
| `test_data/` | 评测用知识库（深度学习概念手册） |

## 核心亮点

1. **混合检索**：向量 + BM25（jieba 分词）→ RRF 融合 → CrossEncoder 重排
2. **完整评测体系**：严格召回率 / 答案正确率 / 反幻觉率 / LLM 忠实率，混合评测（关键词 + LLM-as-Judge）
3. **手写 Agent**：function calling + 多步循环 + AST 白名单安全计算
4. **MCP 封装**：把 RAG 封装成 MCP server，供任意客户端调用

## 评测结果（实测）

| 指标 | 数值 |
|------|------|
| 检索召回率（严格：答案排第 1） | 90% |
| 答案正确率 | 90% |
| 反幻觉率 | 100% |
| LLM 忠实率 | 90% |

重排消融实验：无重排 50% → 有重排 100%。

## 快速开始

1. 安装依赖：`pip install -r requirements.txt`
2. 配置 `.env`：写入 `DEEPSEEK_API_KEY=你的key`
3. 导入知识库：`python import_my_docs.py`
4. 跑评测：`python eval_rag.py`
5. 启动网页：`python web_ui.py`
