# rag_mcp_server.py：把 RAG 封装成 MCP server，暴露给 Cline 等 MCP 客户端
# 复用 06_rag_qa_system 里的 SimpleRAG（rag_system.py 在上两级目录）
import os
import sys
import contextlib
from mcp.server.fastmcp import FastMCP

# 把 06_rag_qa_system 目录加进 import 搜索路径，才能 import 到 rag_system
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from rag_system import SimpleRAG

# 初始化 MCP server（名字会作为工具前缀显示给 Cline）
mcp = FastMCP('rag-knowledge-qa')

# 模块级加载一次 RAG（连 ChromaDB + 加载 embedding 模型），Cline 多次调用复用同一个实例
rag = SimpleRAG()
# rag.enable_rerank()   # 想开重排再取消注释（本地加载 1GB 模型，检索更准但启动变慢）


@mcp.tool()
def search_knowledge_base(question: str) -> str:
    """在个人知识库里检索，返回最相关片段（含来源）。"""
    docs, metas = rag.retrieve(question)
    if not docs:
        return '知识库里没有找到相关内容。'
    return '\n\n'.join(f"[来源:{m['source']}] {d}" for d, m in zip(docs, metas))


@mcp.tool()
def ask_knowledge_base(question: str) -> str:
    """基于个人知识库回答（完整 RAG：检索 + 生成）。"""
    # SimpleRAG.ask 内部会 print 检索日志；stdio 协议占用了 stdout，导到 stderr 避免污染协议消息
    with contextlib.redirect_stdout(sys.stderr):
        return rag.ask(question)


if __name__ == '__main__':
    mcp.run(transport='stdio')   # stdio：Cline 通过标准输入输出和这个 server 通信
