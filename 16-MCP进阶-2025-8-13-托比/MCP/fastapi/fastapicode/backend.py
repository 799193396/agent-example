from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx
import os
from pathlib import Path
import uvicorn

# 获取当前文件所在目录
current_dir = Path(__file__).parent

# 1. 创建FastAPI应用实例
app = FastAPI(title="Deepseek Chat API", version="1.0.0")

# 2. 设置CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. 定义请求体模型
class ChatRequest(BaseModel):
    message: str

# 4. 定义核心的聊天端点
@app.post("/chat")
async def chat_with_ai(request: ChatRequest):
    """
    接收用户消息，调用Deepseek API，返回AI的回复。
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="DEEPSEEK_API_KEY environment variable not set")

    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": request.message}
        ],
        "stream": False
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=data, headers=headers, timeout=30.0)
            response.raise_for_status()
            result = response.json()
            ai_message = result["choices"][0]["message"]["content"]
            return {"reply": ai_message}
        except httpx.RequestError as e:
            raise HTTPException(status_code=500, detail=f"Request to Deepseek API failed: {str(e)}")
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"Deepseek API error: {e.response.text}")
        except (KeyError, IndexError) as e:
            raise HTTPException(status_code=500, detail=f"Unexpected response format from Deepseek API: {str(e)}")

# 5. 提供前端页面
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    # 尝试多种可能的文件路径
    possible_paths = [
        current_dir / "index.html",
        Path.cwd() / "index.html",
        Path(__file__).parent / "index.html"
    ]
    
    for html_path in possible_paths:
        if html_path.exists():
            try:
                with open(html_path, "r", encoding="utf-8") as f:
                    return HTMLResponse(content=f.read())
            except Exception as e:
                continue
    
    # 如果找不到文件，返回一个简单的页面
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Deepseek AI Chat</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            input, button { padding: 10px; margin: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Deepseek AI Chat</h1>
            <p>Frontend file not found. Please make sure index.html exists in the same directory as the Python script.</p>
            <div>
                <input type="text" id="message" placeholder="Type your message..." style="width: 70%">
                <button onclick="sendMessage()">Send</button>
            </div>
            <div id="response" style="margin-top: 20px; padding: 10px; border: 1px solid #ccc; min-height: 100px;"></div>
        </div>
        <script>
            async function sendMessage() {
                const message = document.getElementById('message').value;
                const responseDiv = document.getElementById('response');
                
                if (!message) return;
                
                responseDiv.innerHTML = 'Sending...';
                
                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ message: message })
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        responseDiv.innerHTML = `<strong>AI Response:</strong><br>${data.reply}`;
                    } else {
                        responseDiv.innerHTML = `<strong>Error:</strong> ${data.detail}`;
                    }
                } catch (error) {
                    responseDiv.innerHTML = `<strong>Network Error:</strong> Could not connect to the server.`;
                    console.error('Error:', error);
                }
                
                document.getElementById('message').value = '';
            }
            
            // Allow pressing Enter to send message
            document.getElementById('message').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });
        </script>
    </body>
    </html>
    """)

# 6. 启动服务器
if __name__ == "__main__":
    print("Starting server on http://localhost:8000")
    print("If you can't access the page, try http://127.0.0.1:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)