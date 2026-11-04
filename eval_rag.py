# eval_rag.py：给 RAG 做评测，量化「检索准不准、答得对不对、有没有幻觉」
from rag_system import SimpleRAG, client   # 复用 rag_system 的 client（同一个 key），别另配占位符
import os

# 0. 先导入一份「干净」的评测素材（纯知识、无代码块，答案埋在正文里）
rag = SimpleRAG(db_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chroma_db_eval'))   # 评测用独立库，别和生产库 chroma_db 混（否则 reset 会洗掉部署数据）
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data', '深度学习概念手册.md')
with open(DATA_FILE, encoding='utf-8') as f:
    content = f.read()
rag.reset()
rag.add_documents([content], source='深度学习概念手册.md')
rag.enable_rerank()

# 1. 测试集：人工准备一批「问题 → 期望」的题（这就是「标注数据」，面试能讲）
#    can_answer=True  → 正文里有答案，expect_keyword 是答案所在片段的关键词
#    can_answer=False → 正文里没有，正确行为是「无法回答」（考它编不编造）
TESTS = [
    {'question': '监督学习和无监督学习有什么区别？', 'can_answer': True, 'expect_keyword': '标签'},
    {'question': '过拟合的表现是什么？', 'can_answer': True, 'expect_keyword': '泛化'},
    {'question': '学习率太大会有什么后果？', 'can_answer': True, 'expect_keyword': '震荡'},
    {'question': 'Adam 优化器有什么特点？', 'can_answer': True, 'expect_keyword': 'Adam'},
    {'question': 'ReLU 激活函数为什么常用？', 'can_answer': True, 'expect_keyword': 'ReLU'},
    {'question': '梯度消失怎么缓解？', 'can_answer': True, 'expect_keyword': '残差'},
    {'question': 'Transformer 靠什么建模序列？', 'can_answer': True, 'expect_keyword': '自注意力'},
    {'question': 'RNN 的缺点是什么？', 'can_answer': True, 'expect_keyword': '长距离'},
    {'question': '迁移学习通常怎么做？', 'can_answer': True, 'expect_keyword': '微调'},
    {'question': '语义分割评估用什么指标？', 'can_answer': True, 'expect_keyword': '平均交并比'},
    # ↓ 下面这些正文里没有 → 正确答案是「无法回答」
    {'question': '什么是量子计算的叠加态？', 'can_answer': False, 'expect_keyword': ''},
    {'question': '强化学习的策略梯度怎么实现？', 'can_answer': False, 'expect_keyword': ''},
    {'question': '图神经网络 GNN 适合什么任务？', 'can_answer': False, 'expect_keyword': ''},
]

# 反幻觉判断的匹配词集合：模型说这些，都算「没编造」（避免「没有找到相关内容」被误判成幻觉）
NO_ANSWER_KEYWORDS = ['无法回答', '不知道', '没有找到', '无相关', '未找到', '不存在']

def generate_answer(question, docs, metas):
    """用「给定的检索结果」生成答案，并返回实际喂给模型的上下文。
    评测专用：固定 prompt、不走多轮历史，保证「生成」和「裁判」看同一份上下文。"""
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
        return resp.choices[0].message.content, context   # 返回答案 + 它实际看到的上下文
    except Exception as e:
        return None, context   # 生成失败返回 None，由调用方单独计数——别把「答错」和「没生成出来」混为一谈

# 两个裁判 prompt 只有措辞不同，抽成字典、用一个函数复用（避免复制粘贴两遍）
JUDGE_PROMPTS = {
    'faithful': '''你是评测裁判，判断下面的「模型回答」是否忠于「参考资料」。

【问题】{question}

【参考资料】{context}

【模型回答】{answer}

只回答两个字：回答正确且完全基于参考资料 → 「忠实」；答错或编造了参考资料没有的内容 → 「不忠实」。''',
    'hallucination': '''你是评测裁判，判断下面的「模型回答」是否编造了参考资料里没有的内容。

【问题】{question}

【参考资料】{context}

【模型回答】{answer}

只回答两个字：老实说无法回答、没有编造 → 「无幻觉」；编造了参考资料没有的内容 → 「有幻觉」。''',
}

def llm_judge(question, context, answer, judge_type='faithful'):
    """让大模型当裁判。judge_type='faithful' 判忠实（返回忠实/不忠实），'hallucination' 判幻觉（返回无幻觉/有幻觉）"""
    prompt = JUDGE_PROMPTS[judge_type].format(question=question, context=context, answer=answer)
    try:
        resp = client.chat.completions.create(
            model='deepseek-chat',
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0,   # 裁判要确定，不要随机
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f'裁判失败：{e}'

def evaluate():
    can_answer = [t for t in TESTS if t['can_answer']]
    cannot_answer = [t for t in TESTS if not t['can_answer']]

    log_lines = []   # 收集所有输出，最后写文件

    def log(msg=''):
        print(msg)
        log_lines.append(msg)

    # --- 指标 1：检索召回（严格版：答案关键词必须出现在「第 1 名」片段，才算召回） ---
    recall_hit = 0
    retrieved = {}   # 存每题的检索结果 (docs, metas)，指标 2+4 复用同一份，避免二次检索
    log('\n【检索召回】')
    for t in can_answer:
        docs, metas = rag.retrieve(t['question'])   # 用真实检索（混合召回 + 重排），评的才是实际效果
        retrieved[t['question']] = (docs, metas)    # 保存，后面生成答案、裁判都用这份，保证上下文一致
        # 严格判断：答案必须排第 1；顺便看它实际排第几名
        rank = next((i for i, d in enumerate(docs, 1) if t['expect_keyword'] in d), None)
        hit = (rank == 1)
        if hit:
            recall_hit += 1
        mark = '✅ 第1名' if hit else (f'🔶 第{rank}名' if rank else '❌ 漏召回')
        log(f'  {mark} {t["question"]}（期望词：{t["expect_keyword"]}）')
    recall = recall_hit / len(can_answer)

    # --- 指标 2 + 4：答案正确率（关键词）+ LLM 忠实率（大模型裁判） ---
    answer_correct = 0
    faithful = 0
    judge_fail = 0                # 忠实裁判失败次数
    hallucination_judge_fail = 0  # 幻觉裁判失败次数（和忠实裁判分开，方便定位谁不稳）
    generation_fail = 0
    answered = 0   # 成功生成答案的题数，作为正确率/忠实率的分母（生成失败的题不算）
    log('\n【答案正确率（关键词）＋ LLM 忠实率（裁判）】')
    for t in can_answer:
        docs, metas = retrieved[t['question']]      # 复用指标 1 的检索结果，不查第二次
        ans, context = generate_answer(t['question'], docs, metas)   # 答案和上下文来自同一次检索
        if ans is None:   # 生成失败（API 报错）：不当「答错」、也不送裁判，单独计数跳过
            generation_fail += 1
            log(f'  ⚠️ 生成失败：{t["question"]}（跳过，不计入指标）')
            continue
        answered += 1
        kw_hit = t['expect_keyword'] in ans
        if kw_hit:
            answer_correct += 1
        # 裁判看的 context 就是生成答案时实际喂进去的那份，保证「判」的就是「答」所依据的
        verdict = llm_judge(t['question'], context, ans)
        # 用 in 判断：裁判可能带书名号/空格/Markdown（「忠实」、**忠实**、忠实。），== 会漏判
        if '不忠实' in verdict:
            pass   # 判「不忠实」：不计入忠实率，也不当裁判失败
        elif '忠实' in verdict:
            faithful += 1
        else:
            judge_fail += 1   # 裁判失败（网络错误、格式异常），单独计数，不计入忠实率
        log(f'  {t["question"]}')
        log(f'    关键词：{"✅" if kw_hit else "❌"} ｜ 裁判：{verdict}')
    accuracy = answer_correct / answered if answered else 0
    faithfulness = faithful / answered if answered else 0

    # --- 指标 3：反幻觉率（关键词粗判 + LLM 裁判精判） ---
    no_hallucination = 0
    answered_neg = 0   # 成功生成答案的反幻觉题数（生成失败的题不算）
    log('\n【反幻觉率】')
    for t in cannot_answer:
        docs, metas = rag.retrieve(t['question'])   # 不能答的题也先检索，再让模型判断「无法回答」
        ans, context = generate_answer(t['question'], docs, metas)   # 固定 prompt、不走历史，和指标 2+4 一致
        if ans is None:   # 生成失败：没答出来就无从判断编不编造，跳过并计数
            generation_fail += 1
            log(f'  ⚠️ 生成失败：{t["question"]}（跳过）')
            continue
        answered_neg += 1
        kw_hit = any(kw in ans for kw in NO_ANSWER_KEYWORDS)   # 关键词粗判：出现「无法回答」等
        # 关键词会漏「无法回答，但 XXX」这种「拒答后又继续编」→ 用 LLM 裁判精判有没有真编造
        verdict = llm_judge(t['question'], context, ans, 'hallucination')
        if '有幻觉' in verdict:
            pass   # 编造了 → 幻觉，不计入
        elif '无幻觉' in verdict:
            no_hallucination += 1
        else:
            hallucination_judge_fail += 1   # 幻觉裁判失败，和忠实裁判分开计数，方便定位谁不稳
        log(f'  {"✅" if "无幻觉" in verdict else "❌ 疑似幻觉"} {t["question"]} → {ans[:50]}')
        log(f'    关键词：{"✅" if kw_hit else "❌"} ｜ 裁判：{verdict}')
    anti_hallucination = no_hallucination / answered_neg if answered_neg else 0

    log('\n' + '=' * 40)
    log(f'📊 检索召回率：{recall:.0%}（{recall_hit}/{len(can_answer)}）')
    log(f'📊 答案正确率：{accuracy:.0%}（{answer_correct}/{answered}）')
    log(f'📊 反幻觉率：{anti_hallucination:.0%}（{no_hallucination}/{answered_neg}）')
    log(f'📊 LLM 忠实率：{faithfulness:.0%}（{faithful}/{answered}）')
    if generation_fail:
        log(f'   （生成失败 {generation_fail} 次，未计入正确率/忠实率/反幻觉率）')
    if judge_fail:
        log(f'   （忠实裁判失败 {judge_fail} 次，未计入）')
    if hallucination_judge_fail:
        log(f'   （幻觉裁判失败 {hallucination_judge_fail} 次，未计入）')
    log('=' * 40)

    # 写日志文件，方便留存结果
    with open('eval_result.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))
    log('\n💾 结果已保存到 eval_result.txt')

if __name__ == '__main__':
    evaluate()