# langchain_rag.py：用 LangChain 重写 RAG，逐行对应手写版
import os
# 从项目根目录 .env 读 DEEPSEEK_API_KEY（不硬编码、不进 git）
_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if os.path.exists(_ENV_PATH):
    with open(_ENV_PATH, encoding='utf-8') as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                _k, _v = _line.split('=', 1)
                os.environ.setdefault(_k.strip(), _v.strip())
import sys
import logging
import shutil
import jieba
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
# 混合检索 + 重排：LangChain 1.0 把这些高级检索器挪进了 classic / community 包
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

sys.stdout.reconfigure(encoding='utf-8')            # Windows 中文输出
logging.basicConfig(level=logging.INFO)

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data', '深度学习概念手册.md')

# 1. LLM 指向 DeepSeek（OpenAI 兼容）
llm = ChatOpenAI(model='deepseek-chat', api_key=os.environ.get('DEEPSEEK_API_KEY'), base_url='https://api.deepseek.com')
# 2. Embedding 用本地缓存的模型（和手写版同一个）
embedder = HuggingFaceEmbeddings(model_name='paraphrase-multilingual-MiniLM-L12-v2')
# 3. 切分器（对应手写 split_text）
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# 4. 读概念手册（和手写版 import_my_docs 同一份数据），带 source 溯源
with open(DATA_FILE, encoding='utf-8') as f:
    content = f.read()
raw_docs = [Document(page_content=content, metadata={'source': '深度学习概念手册.md'})]
docs = splitter.split_documents(raw_docs)            # ⬅️ 真正调用切分器（豆包抓的坑）
logging.info('切分后 %d 个 chunk', len(docs))

# 5. 重置 + 建向量库（先清旧库，多次运行不追加，对齐手写 reset()）
shutil.rmtree('./chroma_lc', ignore_errors=True)
vectorstore = Chroma.from_documents(docs, embedder, persist_directory='./chroma_lc')

# 6. 混合检索 + 重排（对应手写两段式 retrieve：向量+BM25→RRF→重排）
vector_retriever = vectorstore.as_retriever(search_kwargs={'k': 10})
bm25_retriever = BM25Retriever.from_documents(docs, k=10, preprocess_func=lambda t: list(jieba.cut(t)))   # 中文要先分词
hybrid = EnsembleRetriever(retrievers=[vector_retriever, bm25_retriever], weights=[0.5, 0.5])             # RRF 融合
reranker = CrossEncoderReranker(model=HuggingFaceCrossEncoder(model_name='D:/bge-reranker-base/models/BAAI--bge-reranker-base/snapshots/master'), top_n=3)
retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=hybrid)

# 7. Prompt 模板（对应手写手拼字符串）
prompt = ChatPromptTemplate.from_template(
    '你是知识库问答助手，只依据参考资料回答，没有答案就说无法回答。\n\n'
    '【参考资料】\n{context}\n\n【问题】\n{question}\n\n【回答】'
)

# 8. LCEL 链：检索 → 拼 prompt → LLM → 解析输出
chain = (
    {'context': retriever, 'question': RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

try:
    answer = chain.invoke('什么是过拟合？')
    print('✅', answer)
except Exception as e:                               # LLM 超时/报错不崩
    logging.error('调用失败：%s', e)

# 9. 复用 eval 的 TESTS 跑全套评测，对比手写版指标（两套实现、同一套测试集）
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

# 9a. 检索召回率：答案关键词排第 1 才算召回（和 compare_rerank 口径一致）
recall_hit = 0
for t in TESTS:
    #  chain.invoke是跑完全套rag,retriver.invoke是只跑检索
    docs = retriever.invoke(t['question'])           # retriever 已含 BM25+RRF+重排，返回 top_n 个 Document
    if docs and t['expect_keyword'] in docs[0].page_content:
        recall_hit += 1
print(f'📊 LangChain 版检索召回率：{recall_hit / len(TESTS):.0%}（手写版 eval 实测 90%）')

# 9b. 答案正确率（关键词）+ 忠实率（LLM 裁判），对应手写版指标 2+4
faith_prompt = PromptTemplate.from_template(
    '判断下面答案是否忠实于参考资料（只在资料里找依据，不编造）。\n'
    '【参考资料】\n{context}\n\n【问题】\n{question}\n\n【答案】\n{answer}\n\n'
    '只回答「忠实」或「不忠实」。'
)
gen = prompt | llm | StrOutputParser()               # 手动拼 context 生成答案（不复用 chain，因为要单独拿 context 给裁判）
judge = faith_prompt | llm | StrOutputParser()
answer_correct = 0                                   # 答案正确率：关键词命中
faithful = 0
gen_fail = 0                                         # 生成失败（网络/API 挂）：不当答错、跳过
judge_fail = 0                                       # 裁判失败：单独计数，不计入忠实率
answered = 0                                         # 成功生成答案的题数，作为正确率/忠实率分母（和手写版一致）
for t in TESTS:
    docs = retriever.invoke(t['question'])
    context = '\n\n'.join(d.page_content for d in docs)
    try:
        answer = gen.invoke({'context': context, 'question': t['question']})
    except Exception as e:
        gen_fail += 1
        logging.error('生成失败：%s', e)
        continue
    answered += 1
    kw_hit = t['expect_keyword'] in answer           # 答案正确率：关键词命中
    if kw_hit:
        answer_correct += 1
    try:
        verdict = judge.invoke({'context': context, 'question': t['question'], 'answer': answer})
    except Exception as e:
        judge_fail += 1
        logging.error('裁判失败：%s', e)
        continue
    if '不忠实' not in verdict:                       # 先判「不忠实」（含「忠实」两字，避免误判）
        faithful += 1
    print(f'  {t["question"]}')                       # 逐条打印，方便定位哪题错/不忠实
    print(f'    关键词：{"✅" if kw_hit else "❌"} ｜ 裁判：{verdict}')
accuracy = answer_correct / answered if answered else 0
faith_rate = faithful / answered if answered else 0
print(f'📊 LangChain 版答案正确率：{accuracy:.0%}（手写版 eval 实测 90%）')
print(f'📊 LangChain 版忠实率：{faith_rate:.0%}（手写版 eval 实测 90%）')
if gen_fail or judge_fail:
    print(f'   （生成失败 {gen_fail} 次、裁判失败 {judge_fail} 次，未计入正确率/忠实率）')

# 9c. 反幻觉率：问知识库里没有的（负面样本），正确行为是「无法回答」，考它编不编造
CANNOT_ANSWER = [
    '什么是量子计算的叠加态？',
    '强化学习的策略梯度怎么实现？',
    '图神经网络 GNN 适合什么任务？',
]
hall_prompt = PromptTemplate.from_template(
    '你是评测裁判，判断下面的「模型回答」是否编造了参考资料里没有的内容。\n'
    '【问题】\n{question}\n\n【参考资料】\n{context}\n\n【模型回答】\n{answer}\n\n'
    '只回答两个字：老实说无法回答、没有编造 → 「无幻觉」；编造了参考资料没有的内容 → 「有幻觉」。'
)
hall_judge = hall_prompt | llm | StrOutputParser()
no_hallucination = 0
hall_gen_fail = 0
hall_judge_fail = 0
answered_neg = 0
for q in CANNOT_ANSWER:
    docs = retriever.invoke(q)
    context = '\n\n'.join(d.page_content for d in docs)
    try:
        answer = gen.invoke({'context': context, 'question': q})
    except Exception as e:
        hall_gen_fail += 1
        logging.error('生成失败：%s', e)
        continue
    answered_neg += 1
    try:
        verdict = hall_judge.invoke({'context': context, 'question': q, 'answer': answer})
    except Exception as e:
        hall_judge_fail += 1
        logging.error('裁判失败：%s', e)
        continue
    if '有幻觉' in verdict:
        mark = '❌ 幻觉'
    elif '无幻觉' in verdict:
        no_hallucination += 1
        mark = '✅ 无幻觉'
    else:
        hall_judge_fail += 1                       # 裁判返回非预期结果，单独计数
        mark = '⚠️ 裁判异常'
    print(f'  {mark}  {q} → {answer[:50]}')          # 逐条打印，方便看哪个问题编造了
    print(f'    裁判：{verdict}')
hall_rate = no_hallucination / answered_neg if answered_neg else 0
print(f'📊 LangChain 版反幻觉率：{hall_rate:.0%}（手写版 eval 实测 100%）')
if hall_gen_fail or hall_judge_fail:
    print(f'   （生成失败 {hall_gen_fail} 次、裁判失败 {hall_judge_fail} 次，未计入反幻觉率）')