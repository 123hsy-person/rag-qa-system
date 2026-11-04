# compare_rerank.py：对比重排前后的检索召回率（严格版：答案排第 1 才算召回）
import os
from rag_system import SimpleRAG

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # 脚本所在目录，路径跟着文件走、不写死

# 和 eval_rag.py 一样的干净素材
DATA_FILE = os.path.join(BASE_DIR, 'test_data', '深度学习概念手册.md')
with open(DATA_FILE, encoding='utf-8') as f:
    content = f.read()

# 能答题（重点看那几个漏召回的）
TESTS = [
    {'question': '监督学习和无监督学习有什么区别？', 'expect_keyword': '标签'},
    {'question': '过拟合的表现是什么？', 'expect_keyword': '泛化'},
    {'question': '学习率太大会有什么后果？', 'expect_keyword': '震荡'},
    {'question': 'Adam 优化器有什么特点？', 'expect_keyword': 'Adam'},
    {'question': 'ReLU 激活函数为什么常用？', 'expect_keyword': 'ReLU'},
    {'question': '梯度消失怎么缓解？', 'expect_keyword': '残差'},
    {'question': 'Transformer 靠什么建模序列？', 'expect_keyword': '自注意力'},
    {'question': 'RNN 的缺点是什么？', 'expect_keyword': '长距离'},
    {'question': '迁移学习通常怎么做？', 'expect_keyword': '微调'},
    {'question': '语义分割评估用什么指标？', 'expect_keyword': '平均交并比'},
]

def fresh_rag():
    """造一个「导入了同一份干净素材」的独立 RAG 实例，隔离各组实验状态"""
    r = SimpleRAG(db_path=os.path.join(BASE_DIR, 'chroma_db_eval'))   # 评测用独立库，别和生产库 chroma_db 混
    r.reset()
    r.add_documents([content], source='深度学习概念手册.md')
    return r

def rank_of(rag, question, expect):
    """返回答案关键词在检索结果里排第几名（没进 top10 返回 None）"""
    docs, _ = rag.retrieve(question, top_k=10)   # 取 10 个拿真实排名，别只取 3（否则第 4~6 名会被标成「漏」）
    return next((i for i, d in enumerate(docs, 1) if expect in d), None)

def measure_recall(rag):
    """严格版召回率：答案排第 1 才算召回"""
    hit = sum(1 for t in TESTS if rank_of(rag, t['question'], t['expect_keyword']) == 1)
    return hit / len(TESTS)

# 输出同时落盘：控制台 + compare_result.txt，实验结论关了程序也不丢
OUTPUT = []
def log(msg=''):
    print(msg)
    OUTPUT.append(msg)

def show(rag, title):
    log(f'\n【{title}】')
    for t in TESTS:
        r = rank_of(rag, t['question'], t['expect_keyword'])
        mark = '✅ 第1' if r == 1 else (f'🔶 第{r}' if r else '❌ 漏')
        log(f'  {mark}  {t["question"]}')

# 1. 无重排（独立实例 A）
rag_no_rerank = fresh_rag()
show(rag_no_rerank, '无重排')
recall_before = measure_recall(rag_no_rerank)
log(f'  → 召回率 {recall_before:.0%}')

# 2. 有重排（独立实例 B：重排开关只在 B 上开，不污染 A）
#    enable_rerank() 只有「开」没有「关」，同一个实例开一次就永久生效；
#    所以必须新建实例隔离，就算调换顺序（先测有重排）对照组也不会失效
rag_rerank = fresh_rag()
rag_rerank.enable_rerank()
show(rag_rerank, '有重排')
recall_after = measure_recall(rag_rerank)
log(f'  → 召回率 {recall_after:.0%}')

log(f'\n========== 变化：{recall_before:.0%} → {recall_after:.0%}（{recall_after - recall_before:+.0%}）==========')

with open(os.path.join(BASE_DIR, 'compare_result.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(OUTPUT))
print(f'📄 结果已写入 compare_result.txt')
