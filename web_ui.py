from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

from agent import Agent

app = FastAPI()
agent= Agent()


HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>我的知识库</title>
    <meta charset="utf-8">
    <style>
        body { max-width: 800px; margin: 50px auto; font-family: sans-serif; }
        #chat { border: 1px solid #ddd; padding: 20px; min-height: 400px; max-height: 600px; overflow-y: auto; }
        .q { color: blue; margin: 10px 0; }
        .a { color: green; margin: 10px 0 20px 0; border-bottom: 1px solid #eee; padding-bottom: 10px; }
        input[type="text"] { width: 80%; padding: 10px; }
        button { padding: 10px 20px; }
    </style>
</head>
<body>
    <h1>📚 我的知识库</h1>
    <p>基于笔记做问答</p>
    <div id="chat"></div>
    <input type="text" id="question" placeholder="问我任何问题..." onkeypress="if(event.key==='Enter')ask()">
    <button onclick="ask()">提问</button>
    <button onclick="clearChat()">清空对话</button>

    <script>
        function ask() {
            const q = document.getElementById('question').value;
            if (!q) return;
            document.getElementById('chat').innerHTML += `<div class="q">🙋 ${q}</div>`;
            fetch('/api/ask', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({question: q})
            })
            .then(r => r.json())
            .then(data => {
                document.getElementById('chat').innerHTML += `<div class="a">🤖 ${data.answer}</div>`;
                document.getElementById('question').value = '';
            });
        }

        function clearChat() {
            fetch('/api/clear', {method: 'POST'})
            .then(() => { document.getElementById('chat').innerHTML = ''; });
        }
    </script>
</body>
</html>
'''

class Question(BaseModel):
    question: str

@app.get('/', response_class=HTMLResponse)
def index():
    return HTML

@app.post('/api/ask')
def ask(q: Question):
    answer = agent.ask(q.question)   # 换成 Agent 的 ask（走实例方法）
    return {'answer': answer}

@app.post('/api/clear')
def clear():
    agent.clear_history()   # 清空对话历史，回到「只记得系统提示词」的状态
    return {'ok': True}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=5000)