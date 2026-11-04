# debug_badcase.py：针对 eval_rag 里「过拟合的表现」这道 badcase 做分层排查，定位到底哪一层出错
# 排查思路（面试能讲）：RAG 是管道，badcase 要按 ①评测/标注 → ②检索 → ③生成 → ④裁判 四层从上到下二分定位
#   ——先排除「假 badcase」（标注问题），再逐层往管道下游追，定位到哪层就在哪层修
import os
from rag_system import SimpleRAG, client   # 复用 rag_system 的 client（同一个 key）

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, 'test_data', '深度学习概念手册.md')
QUESTION = '过拟合的表现是什么？'
EXPECT = '泛化'

OUTPUT = []   # 结果同时落盘 debug_badcase.txt，关了程序也不丢
def log(msg=''):
    print(msg)
    OUTPUT.append(msg)

# 0. 复现 eval 环境：独立评测库 + 同一份干净素材（别动生产库 chroma_db）
rag = SimpleRAG(db_path=os.path.join(BASE_DIR, 'chroma_db_eval'))
with open(DATA_FILE, encoding='utf-8') as f:
    content = f.read()
rag.reset()
rag.add_documents([content], source='深度学习概念手册.md')

# ============ 第①层：评测/标注对不对（先排除「假 badcase」） ============
log('\n' + '=' * 60)
log('第①层 评测/标注：期望词「泛化」在不在库里？出现在哪些片段？')
log('  → 如果「泛化」压根不在库里，那是标注/测试集选错了词（假 badcase），系统没坏')
for i, d in enumerate(rag.all_docs, 1):
    if EXPECT in d:
        log(f'\n  ★ 片段#{i} 含「{EXPECT}」：\n     {d}')

# ============ 第②层：检索（打印排名，看答案片段排第几） ============
def show_retrieve(title, question, top_k):
    docs, metas = rag.retrieve(question, top_k=top_k)
    rank = next((i for i, d in enumerate(docs, 1) if EXPECT in d), None)
    log(f'\n  [{title}] top{top_k} 里「{EXPECT}」排第 {rank if rank else "无"} 名')
    for i, (d, m) in enumerate(zip(docs, metas), 1):
        mark = '★ 含期望词' if EXPECT in d else ''
        # top3 是生成真正看到的片段，打全文看它为什么赢/输；top10 只看排名，截断省屏幕
        body = d if top_k <= 3 else (d[:120] + '...' if len(d) > 120 else d)
        log(f'    第{i}名 {mark} (来源:{m["source"]})\n       {body}')
    return rank

log('\n' + '=' * 60)
log('第②层 检索：无重排 vs 有重排，答案片段排第几？')
log('  → 如果答案排第 1，检索没问题，往下查生成；排第 2~N 或漏召回，根子在检索')

# 2a. 无重排（还没开 reranker，纯「向量 + BM25 → RRF」混合召回）
show_retrieve('无重排', QUESTION, top_k=10)

# 2b. 有重排（加载本地 bge-reranker，和 eval_rag / compare_rerank 口径一致）
rag.enable_rerank()
# 注意：eval_rag 用默认 top_k=3（所以看到「第2名」），compare_rerank 用 top_k=10（所以看到「第1名」）
#   ——两套脚本数字差一名，多半就是 top_k 口径不同，面试别把两套数字混着报
show_retrieve('有重排（eval_rag 口径 top3）', QUESTION, top_k=3)
show_retrieve('有重排（compare_rerank 口径 top10）', QUESTION, top_k=10)

# ============ 第③层：生成（检索对了还答错，才是生成层问题） ============
def generate_answer(question, docs, metas):
    """和 eval_rag 同款：固定 prompt、不走多轮历史，返回答案 + 实际喂给模型的上下文"""
    context = '\n\n'.join(f"[来源: {m['source']}] {d}" for d, m in zip(docs, metas))
    prompt = f'''你是一个知识库问答助手。请根据以下参考资料回答用户的问题。
如果参考资料中没有答案，就说「根据现有资料无法回答」，不要编造。

【参考资料】
{context}

【用户问题】
{question}

【回答】'''
    try:
        resp = client.chat.completions.create(
            model='deepseek-chat',
            messages=[
                {'role': 'system', 'content': '你是知识库助手，只依据参考资料回答，没有答案就说无法回答，不要编造。'},
                {'role': 'user', 'content': prompt},
            ],
            temperature=0.3,
            max_tokens=1000,
        )
        return resp.choices[0].message.content, context
    except Exception as e:
        return None, context

log('\n' + '=' * 60)
log('第③层 生成：检索结果给模型，答案里有没有「泛化」？')
docs, metas = rag.retrieve(QUESTION, top_k=3)   # 和 eval_rag 一样取 top3
ans, context = generate_answer(QUESTION, docs, metas)
if ans is None:
    log('  ⚠️ 生成失败（API 报错），本题跳过')
else:
    log(f'\n  模型回答：{ans}')
    log(f'  关键词「{EXPECT}」命中：{"✅" if EXPECT in ans else "❌"}')

# ============ 第④层：裁判（判断系统误判，还是裁判 prompt 的问题） ============
FAITHFUL_PROMPT = '''你是评测裁判，判断下面的「模型回答」是否忠于「参考资料」。

【问题】{question}

【参考资料】{context}

【模型回答】{answer}

只回答两个字：回答正确且完全基于参考资料 → 「忠实」；答错或编造了参考资料没有的内容 → 「不忠实」。'''

def llm_judge(question, context, answer):
    """和 eval_rag 同款的忠实裁判"""
    prompt = FAITHFUL_PROMPT.format(question=question, context=context, answer=answer)
    try:
        resp = client.chat.completions.create(
            model='deepseek-chat',
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f'裁判失败：{e}'

log('\n' + '=' * 60)
log('第④层 裁判：让大模型判「忠实 / 不忠实」')
if ans is not None:
    verdict = llm_judge(QUESTION, context, ans)
    log(f'  裁判判定：{verdict}')
    log('  → 裁判判「不忠实」说明它没冤枉系统；若裁判判「忠实」而答案明明错，才是裁判 prompt 的问题')

# ============ 结论：把四层信号合起来，定位到哪一层 ============
log('\n' + '=' * 60)
log('【怎么读这些结果】')
log('  ① 片段里确实有「泛化」→ 排除假 badcase，标注没问题')
log('  ② 答案片段在无重排时掉到第几、有重排后第几 → 决定根子是不是检索')
log('     若第1名是「正则化/数据增强」里带「泛化」的片段、过拟合片段被挤到后面 → 检索没抓住语义')
log('  ③ 检索排第2，模型照着排第1的错片段答，答案自然没有「泛化」→ 生成是被检索连累')
log('  ④ 裁判判「不忠实」→ 裁判没错，它在如实反映检索+生成的偏差')
log('  → 典型结论：这是一个「检索层」badcase（语义近、字面远），修法方向是加重排 / 调切分 / 加关键词混合召回')

with open(os.path.join(BASE_DIR, 'debug_badcase.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(OUTPUT))
print('\n📄 排查结果已写入 debug_badcase.txt')
