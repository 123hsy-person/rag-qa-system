# agent.py：Agent = RAG（查知识库）+ 工具（时间/算数）+ 多步循环 + 多轮历史
from rag_system import SimpleRAG, client   # client 复用 rag_system 的（同一个 key），别另配占位符
import json
import datetime
import ast
import operator

# ============ 工具定义（JSON 描述给模型看，Python 实现真正执行） ============
TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'search_knowledge_base',
            'description': '在个人知识库里检索。用户问「笔记里讲过什么」「什么是 XX」时调用。',
            'parameters': {
                'type': 'object',
                'properties': {'question': {'type': 'string', 'description': '要检索的问题'}},
                'required': ['question'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_current_time',
            'description': '获取当前日期和时间，可指定往后偏移多少分钟。注意：offset_minutes 单位是分钟，1 小时 = 60 分钟。',
            'parameters': {
                'type': 'object',
                'properties': {'offset_minutes': {'type': 'integer', 'description': '往后偏移的分钟数（1 小时 = 60 分钟）', 'default': 0}},
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'calculate',
            'description': '安全计算数学表达式，如「3**10」「128*7」。只支持数字和四则运算。',
            'parameters': {
                'type': 'object',
                'properties': {'expression': {'type': 'string', 'description': '数学表达式'}},
                'required': ['expression'],
            },
        },
    },
]

# ============ 工具实现：无状态的模块级函数（不碰对话历史） ============
def get_current_time(offset_minutes=0):
    t = datetime.datetime.now() + datetime.timedelta(minutes=offset_minutes)
    return t.strftime('%Y-%m-%d %H:%M:%S')

# 安全计算：用 ast 白名单，只放行数字和四则运算，杜绝 eval 的任意代码执行漏洞
_ALLOWED_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg,
}

def calculate(expression):
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
        return str(_eval(ast.parse(expression, mode='eval')))   # 统一返回字符串，避免类型混乱
    except ZeroDivisionError:
        return '计算错误：不能除以 0。'
    except Exception as e:
        return f'计算失败：{e}'

# 系统提示词抽成常量：__init__ 和 clear_history 都要用，避免重复写两遍
SYSTEM_PROMPT = (
    '你是个人 AI 助手。遇到下面的情况，必须调用对应工具，不要自己猜答案：\n'
    '1. 问知识库里的内容（「什么是 XX」「笔记讲过什么」）→ 调 search_knowledge_base\n'
    '2. 问时间（「现在几点」「3 小时后几点」）→ 调 get_current_time，注意 offset_minutes 单位是分钟，1 小时 = 60 分钟\n'
    '3. 要算数（「3 的 10 次方」「128 乘 7」）→ 调 calculate\n'
    '拿到工具结果后再回答。'
)

MAX_STEPS = 5   # 最多循环 5 步，防止模型无限调工具烧 token


class Agent:
    """Agent 用 self.messages 存整段对话（含 tool_calls），跨多次 ask 保持多轮记忆。"""

    def __init__(self):
        self.rag = SimpleRAG()   # 复用你的 RAG（建连接、加载模型）
        self.rag.enable_rerank()   # 开启重排：最终部署用上 reranker，和 eval_rag/compare_rerank 口径一致
        self.messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
        # 工具表：名字 → 真正执行的函数。search_knowledge_base 需要 self.rag，所以是方法
        self.tool_funcs = {
            'search_knowledge_base': self.search_knowledge_base,
            'get_current_time': get_current_time,
            'calculate': calculate,
        }

    def search_knowledge_base(self, question):
        docs, metas = self.rag.retrieve(question)   # 复用两段式检索（向量+BM25+重排），和 rag_system 保持一致
        if not docs:   # 兜底：检索为空，别让模型拿空内容去编
            return '知识库里没有找到相关内容。'
        return '\n\n'.join(f"[来源:{m['source']}] {d}" for d, m in zip(docs, metas))

    def ask(self, question):
        self.messages.append({'role': 'user', 'content': question})

        for step in range(MAX_STEPS):
            try:
                resp = client.chat.completions.create(
                    model='deepseek-chat',
                    messages=self.messages,   # 带上前几轮历史，模型能多轮追问
                    tools=TOOLS,
                )
            except Exception as e:
                return f'❌ 调用大模型失败：{e}'

            msg = resp.choices[0].message

            if not msg.tool_calls:          # 模型不调工具了 → 要回答，把回答也记进历史，结束循环
                self.messages.append({'role': 'assistant', 'content': msg.content})
                return msg.content

            self.messages.append(msg)       # 否则执行工具，把结果塞回去，进入下一步
            for tc in msg.tool_calls:
                name = tc.function.name
                print(f'🔧 第 {step+1} 步：调用工具 {name}')   # 日志：能看到 Agent 每一步在干嘛
                try:
                    args = json.loads(tc.function.arguments or '{}')   # 参数解析可能失败
                    print(f'   参数：{args}')
                    fn = self.tool_funcs.get(name)
                    if fn is None:
                        result = f'❌ 未知工具：{name}'
                    else:
                        result = str(fn(**args))                        # 工具执行可能出错
                except Exception as e:
                    result = f'❌ 工具 {name} 执行失败：{e}'
                print(f'   结果：{result}')
                self.messages.append({'role': 'tool', 'tool_call_id': tc.id, 'content': result})

        # 循环用完还没停：强制不带 tools 再问一次，逼它回答
        try:
            final = client.chat.completions.create(model='deepseek-chat', messages=self.messages)
            answer = final.choices[0].message.content
            self.messages.append({'role': 'assistant', 'content': answer})
            return answer
        except Exception as e:
            return f'❌ 调用大模型失败：{e}'

    def clear_history(self):
        """清空对话记忆，回到只有系统提示词的状态，开始新一轮对话。"""
        self.messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]


