import os
from rag_system import SimpleRAG
# import_my_docs.py
# 知识库数据源 = 深度学习概念手册（和 eval_rag / compare_rerank 同一份，三处口径一致）
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data', '深度学习概念手册.md')

rag = SimpleRAG()   # 默认连生产库 chroma_db（部署 web_ui / agent 读的就是这个库）
rag.reset()         # 清空旧数据（含之前误入的笔记），从头导入
with open(DATA_FILE, encoding='utf-8') as f:
    content = f.read()
rag.add_documents([content], source='深度学习概念手册.md')
print('✅ 知识库已导入：深度学习概念手册.md')
