# rag_system.py
import os

# 从项目根目录 .env 读 DEEPSEEK_API_KEY（不硬编码、不进 git；.env 已加进 .gitignore）
_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if os.path.exists(_ENV_PATH):
    with open(_ENV_PATH, encoding='utf-8') as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                _k, _v = _line.split('=', 1)
                os.environ.setdefault(_k.strip(), _v.strip())

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'   # 下载 HF 模型走 Clash 代理
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
os.environ['NO_PROXY'] = 'api.deepseek.com'          # DeepSeek 是国内服务，不走代理

from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import chromadb
from sentence_transformers import SentenceTransformer
import hashlib
import jieba
from rank_bm25 import BM25Okapi


client = OpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'), base_url='https://api.deepseek.com')

class SimpleRAG:
    def __init__(self, db_path=None):
        # 连接 ChromaDB（数据存本地文件）。评测/对比传独立路径（chroma_db_eval），避免 reset 洗掉生产库
        self.chroma = chromadb.PersistentClient(
            path=db_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chroma_db')
        )
        self.collection = self.chroma.get_or_create_collection(name='my_documents')

        # Embedding 模型：把文字变成向量（第一次跑会下载 ~100MB）
        self.embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

        # 多轮对话记忆：保存历史对话，才能连续追问
        self.history = []

        # 1. __init__ 里加一个切分器对象（注意：这个 split_text 是库方法，跟你手写的 def split_text 是两个东西）
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=['\n\n', '\n', '。', '！', '？', ' ', '']  # 逐级降级切
        )

        # 1. __init__ 里加：重排模型，默认不加载（None = 不重排，省得每次启动下 1GB 模型）
        self.reranker = None
        self.bm25 = None

        # 连库时如果库里已有数据，自动建 BM25 索引，让检索默认就是「向量 + BM25」混合
        if self.collection.count() > 0:
            self.build_bm25()

    # ---------- 切分：段落 → 句子，逐级切 ----------


    def build_bm25(self):
        """把库里所有文档建成 BM25 关键词索引（中文要先分词）"""
        data = self.collection.get()
        self.all_docs = data['documents']
        self.all_metas = data['metadatas']   # 顺便存下来源，供检索溯源
        self.bm25 = BM25Okapi([list(jieba.cut(d)) for d in self.all_docs])

    # ---------- 添加文档：增量 + 元数据溯源 ----------
    def add_documents(self, documents, source='未知'):
        """增量添加：已经在库里的自动跳过，只加新的。source=来源文件名"""
        existing_ids = set(self.collection.get()['ids'])

        new_chunks, new_ids = [], []
        for doc in documents: # 可迭代，documents得是列表
            parts = self.splitter.split_text(doc)
            for part in parts:
                cid = hashlib.md5(part.encode('utf-8')).hexdigest()
                if cid not in existing_ids:      # 只加库里没有的，实现增量
                    new_chunks.append(part)
                    new_ids.append(cid)
                    existing_ids.add(cid)

        if not new_chunks:
            print('⚠️ 没有新增内容，全部已存在')
            return

        embeddings = self.embedder.encode(new_chunks).tolist()
        self.collection.add(
            embeddings=embeddings,
            documents=new_chunks,
            ids=new_ids,
            metadatas=[{'source': source} for _ in new_chunks]   # 记录来源，方便溯源
        )
        print(f'✅ 新增 {len(new_chunks)} 个片段入库（重复的已跳过）')
        self.build_bm25()

    # 2. 加一个开关方法 + 一个统一的检索方法：
    def enable_rerank(self):
        """启用重排。用本地已下载的模型（ModelScope 下好的），不联网。"""
        from sentence_transformers import CrossEncoder
        self.reranker = CrossEncoder('D:/bge-reranker-base/models/BAAI--bge-reranker-base/snapshots/master')

    def retrieve(self, question, top_k=3):
        # 两段式检索：① 混合召回（向量 + BM25 → RRF）② 重排精排。返回 (docs, metas)
        # 第一段：召回（粗排）
        q_emb = self.embedder.encode(question).tolist()
        vec_res = self.collection.query(query_embeddings=[q_emb], n_results=top_k * 2,
                                        include=['documents', 'metadatas'])
        vec_docs, vec_metas = vec_res['documents'][0], vec_res['metadatas'][0]

        bm25_docs = []
        if self.bm25 is not None:  # 没建 BM25 就退回纯向量
            scores = self.bm25.get_scores(list(jieba.cut(question)))
            bm25_docs = [d for _, d in sorted(zip(scores, self.all_docs), reverse=True)[:top_k * 2]]

        fused = {}  # RRF 融合：只看排名不看分数
        for rank, d in enumerate(vec_docs):
            fused[d] = fused.get(d, 0) + 1 / (60 + rank)
        for rank, d in enumerate(bm25_docs):
            fused[d] = fused.get(d, 0) + 1 / (60 + rank)
        candidates = sorted(fused, key=fused.get, reverse=True)[:top_k * 2]

        # 候选对应的来源：优先用全库映射（BM25 召回也能溯源），没建 BM25 退回向量路
        if self.bm25 is not None:
            doc2meta = dict(zip(self.all_docs, self.all_metas))
        else:
            doc2meta = dict(zip(vec_docs, vec_metas))
        metas = [doc2meta.get(d, {'source': '未知'}) for d in candidates]

        # 第二段：精排（重排）
        if self.reranker is not None:
            pairs = [[question, d] for d in candidates]
            scores = self.reranker.predict(pairs)
            ranked = sorted(zip(scores, candidates, metas), key=lambda x: x[0], reverse=True)
            candidates = [d for _, d, _ in ranked[:top_k]]
            metas = [m for _, _, m in ranked[:top_k]]
        else:
            candidates = candidates[:top_k]
            metas = metas[:top_k]

        return candidates, metas

    # ---------- 提问：多轮 + 截断 + 存原文 + 异常捕获 ----------
    def ask(self, question, top_k=3, max_history=6):

        # Step 1：检索（连元数据一起取；启用了 reranker 会重排）
        retrieved_docs, metas = self.retrieve(question, top_k)
        context = '\n\n'.join(f"[来源: {m['source']}] {d}" for d, m in zip(retrieved_docs, metas))

        # Step 2：构造当前问题的 prompt
        prompt = f"""你是一个知识库问答助手。请根据以下参考资料回答用户的问题。
如果参考资料中没有答案，就说「根据现有资料无法回答」，不要编造。

【参考资料】
{context}

【用户问题】
{question}

【回答】"""

        # Step 3：只取最近 max_history 条历史，防止越聊越长超 token 上限
        recent = self.history[-max_history:]
        messages = [
            {'role': 'system', 'content': '你是知识库助手，只依据参考资料回答，没有答案就说无法回答，不要编造。'},
        ]
        messages += recent                                    # 之前的对话
        messages += [{'role': 'user', 'content': prompt}]     # 当前问题

        # Step 4：调大模型（异常捕获：网络/API 出错不崩溃）
        try:
            response = client.chat.completions.create(
                model='deepseek-chat',
                messages=messages,
                temperature=0.3,  # 低温度 = 更确定、少编造
                max_tokens=1000
            )
            answer = response.choices[0].message.content
        except Exception as e:
            print(f'❌ 调用大模型失败：{e}')
            return '抱歉，服务出错了，请稍后再试。'

        # Step 5：把「问题 + 检索到的原文」和「回答」一起记进历史
        #    长距离追问时，模型还记得之前的原始资料
        self.history.append({'role': 'user', 'content': f'{question}\n\n【参考资料】\n{context}'})
        self.history.append({'role': 'assistant', 'content': answer})

        # 打印参考了哪些片段（含来源，方便溯源调试）
        print(f'\n📚 参考了 {len(retrieved_docs)} 个片段:')
        for i, (d, m) in enumerate(zip(retrieved_docs, metas)):
            print(f'  [{i+1}] (来源:{m["source"]}) {d[:80]}...')

        return answer

    def clear_history(self):
        """清空对话记忆，开始新一轮对话。"""
        self.history = []

    def reset(self):
        """清空向量库（想重新导入时先调用）。"""
        ids = self.collection.get()['ids']
        if ids:
            self.collection.delete(ids=ids)
        self.bm25 = None      # 清空 BM25 索引，避免读到已删除的文档
        self.all_docs = None  # 同步清空文档缓存
        print('✅ 向量库已清空')




