# agent_graph.py：用 LangGraph 把 agent.py 的手写循环重写成 StateGraph
import os
import datetime
import ast
import operator
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from rag_system import SimpleRAG   # 复用 RAG（和 agent.py 一样）

# 先建 RAG 实例：下面的 search_knowledge_base 工具要闭包用它；和 agent.py 的 __init__ 一样（建 RAG + 开重排）
rag = SimpleRAG()
rag.enable_rerank()

# ---------- 工具：把 agent.py 的 TOOLS + 实现，换成 @tool 装饰器 ----------

@tool
def search_knowledge_base(question: str) -> str:
    """在个人知识库里检索。用户问「笔记里讲过什么」「什么是 XX」时调用。"""
    docs, metas = rag.retrieve(question)
    if not docs:
        return '知识库里没有找到相关内容。'
    return '\n\n'.join(f"[来源:{m['source']}] {d}" for d, m in zip(docs, metas))

@tool
def get_current_time(offset_minutes: int = 0) -> str:
    """获取当前日期和时间，可指定往后偏移多少分钟（1 小时 = 60 分钟）。"""
    t = datetime.datetime.now() + datetime.timedelta(minutes=offset_minutes)
    return t.strftime('%Y-%m-%d %H:%M:%S')

# 安全计算：ast 白名单，和 agent.py 完全一样
_ALLOWED_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg,
}

@tool
def calculate(expression: str) -> str:
    """安全计算数学表达式，如「3**10」「128*7」。只支持数字和四则运算。"""
    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
            return _ALLOWED_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
            return _ALLOWED_OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f'不支持的表达式：{ast.dump(node)}')
    try:
        return str(_eval(ast.parse(expression, mode='eval')))
    except ZeroDivisionError:
        return '计算错误：不能除以 0。'
    except Exception as e:
        return f'计算失败：{e}'

tools = [search_knowledge_base, get_current_time, calculate]

SYSTEM_PROMPT = (
    '你是个人 AI 助手。遇到下面的情况，必须调用对应工具，不要自己猜答案：\n'
    '1. 问知识库里的内容（「什么是 XX」「笔记讲过什么」）→ 调 search_knowledge_base\n'
    '2. 问时间（「现在几点」「3 小时后几点」）→ 调 get_current_time，注意 offset_minutes 单位是分钟，1 小时 = 60 分钟\n'
    '3. 要算数（「3 的 10 次方」「128 乘 7」）→ 调 calculate\n'
    '拿到工具结果后再回答。'
)

# ---------- State：LangGraph 的「共享状态」，对应手写版 self.messages ----------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # add_messages 是 reducer：新消息自动追加进列表

# ---------- 节点 ----------
llm = ChatOpenAI(
    model='deepseek-chat',
    api_key=os.environ.get('DEEPSEEK_API_KEY', 'sk-YOUR-KEY'),   # 换成你的 key 或设环境变量
    base_url='https://api.deepseek.com',
).bind_tools(tools)   # bind_tools = 手写版 client.chat.completions.create(..., tools=TOOLS)

def call_model(state: AgentState):
    # 不带装饰器给人看
    """调一次 LLM，把 AI 回复追加进 messages。对应手写循环里的一次 chat.completions.create。"""
    return {'messages': [llm.invoke(state['messages'])]}

# ---------- 条件边：判断「继续调工具」还是「结束回答」 ----------
def should_continue(state: AgentState):
    """看最后一条消息有没有 tool_calls。对应手写版 if not msg.tool_calls 那行。"""
    last = state['messages'][-1]
    if getattr(last, 'tool_calls', None):   # 有 tool_calls → 去执行工具；没有 → 结束
        return 'tools'
    return 'end'

# ---------- 组装图：循环 = 条件边 + 回边 ----------
graph = StateGraph(AgentState)
graph.add_node('call_model', call_model)
graph.add_node('tools', ToolNode(tools))          # ToolNode = 手写版「执行工具」那几行
graph.add_edge(START, 'call_model')               # 从 call_model 开始
graph.add_conditional_edges('call_model', should_continue, {'tools': 'tools', 'end': END})
graph.add_edge('tools', 'call_model')             # 工具执行完回到 call_model = 循环
app = graph.compile()

if __name__ == '__main__':
    # 跑一轮：system 提示词 + 用户问题
    result = app.invoke({
        'messages': [SystemMessage(content=SYSTEM_PROMPT),
                     HumanMessage(content='3 小时 20 分钟后几点？')],
    }, config={'recursion_limit': 6})   # recursion_limit = 手写版 MAX_STEPS 的兜底
    print(result['messages'][-1].content)