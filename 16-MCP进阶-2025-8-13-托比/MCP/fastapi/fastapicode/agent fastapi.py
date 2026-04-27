
import asyncio
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import aiofiles
import httpx

# 确保数据目录存在
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "system.db"

class Config:
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "your-deepseek-api-key-here")
    DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {'.txt', '.py', '.js', '.h'
                                                'tml', '.css', '.json', '.md', '.csv'}


class TaskRequest(BaseModel):
    agent_type: str
    task: str
    files: Optional[List[str]] = []
    params: Optional[Dict[str, Any]] = {}

class AgentResponse(BaseModel):
    success: bool
    result: str
    files_created: List[str] = []
    data: Optional[Dict[str, Any]] = None

class DeepseekClient:
    def __init__(self):
        self.api_key = Config.DEEPSEEK_API_KEY
        self.base_url = Config.DEEPSEEK_BASE_URL
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def chat_completion(self, messages: List[Dict], model: str = "deepseek-chat"):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 4000
        }
        
        try:
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                return f"API错误: {response.status_code} - {response.text}"
        except Exception as e:
            return f"请求失败: {str(e)}"

class MCPFileSystem:
    def __init__(self, root_path: str = "data"):
        self.root_path = Path(root_path)
        self.root_path.mkdir(exist_ok=True)
    
    async def read_file(self, filepath: str) -> str:
        """读取文件内容"""
        try:
            full_path = self.root_path / filepath
            if not full_path.exists():
                return f"文件不存在: {filepath}"
            
            async with aiofiles.open(full_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            return content
        except Exception as e:
            return f"读取文件失败: {str(e)}"
    
    async def write_file(self, filepath: str, content: str) -> str:
        """写入文件内容"""
        try:
            full_path = self.root_path / filepath
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            async with aiofiles.open(full_path, 'w', encoding='utf-8') as f:
                await f.write(content)
            return f"文件已保存: {filepath}"
        except Exception as e:
            return f"写入文件失败: {str(e)}"
    
    async def list_files(self, directory: str = ".") -> List[Dict]:
        """列出目录下的文件"""
        try:
            dir_path = self.root_path / directory
            if not dir_path.exists():
                return []
            
            files = []
            for item in dir_path.iterdir():
                files.append({
                    "name": item.name,
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else 0,
                    "modified": datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                })
            return files
        except Exception as e:
            return [{"error": f"列出文件失败: {str(e)}"}]


class MCPDatabase:
    def __init__(self, db_path: str = str(DB_PATH)):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建任务记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_type TEXT NOT NULL,
                task TEXT NOT NULL,
                result TEXT,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            )
        ''')
        
        # 创建文件记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                size INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def execute_query(self, query: str, params: tuple = ()) -> List[Dict]:
        """执行SQL查询"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute(query, params)
            
            if query.strip().upper().startswith('SELECT'):
                rows = cursor.fetchall()
                result = [dict(row) for row in rows]
            else:
                conn.commit()
                result = [{"affected_rows": cursor.rowcount}]
            
            conn.close()
            return result
        except Exception as e:
            return [{"error": f"数据库操作失败: {str(e)}"}]
    
    async def log_task(self, agent_type: str, task: str, result: str = None, status: str = "completed"):
        """记录任务执行"""
        query = "INSERT INTO tasks (agent_type, task, result, status, completed_at) VALUES (?, ?, ?, ?, ?)"
        params = (agent_type, task, result, status, datetime.now().isoformat())
        await self.execute_query(query, params)

class BaseAgent:
    def __init__(self, deepseek_client: DeepseekClient, mcp_fs: MCPFileSystem, mcp_db: MCPDatabase):
        self.deepseek = deepseek_client
        self.mcp_fs = mcp_fs
        self.mcp_db = mcp_db
    
    async def process(self, task: str, files: List[str] = [], params: Dict = {}) -> AgentResponse:
        raise NotImplementedError

class CodeAgent(BaseAgent):
    """代码生成和分析代理"""
    
    async def process(self, task: str, files: List[str] = [], params: Dict = {}) -> AgentResponse:
        try:
            # 读取相关文件内容
            file_contents = {}
            for file in files:
                content = await self.mcp_fs.read_file(file)
                file_contents[file] = content
            
            # 构建提示词
            messages = [
                {"role": "system", "content": "你是一个专业的代码助手，能够生成、分析和优化代码。请根据用户需求生成高质量的代码。"},
                {"role": "user", "content": f"""
任务: {task}

相关文件内容:
{json.dumps(file_contents, ensure_ascii=False, indent=2)}

请生成或分析代码，并提供详细说明。如果需要创建新文件，请明确指出文件名和内容。
"""}
            ]
            
            result = await self.deepseek.chat_completion(messages)
            
            # 如果结果包含代码文件，自动保存
            created_files = []
            if "```" in result and ("文件名:" in result or "保存为:" in result):
                # 简单的文件提取逻辑
                lines = result.split('\n')
                current_file = None
                current_content = []
                
                for line in lines:
                    if "文件名:" in line or "保存为:" in line:
                        if current_file and current_content:
                            content = '\n'.join(current_content)
                            await self.mcp_fs.write_file(current_file, content)
                            created_files.append(current_file)
                        
                        current_file = line.split(':')[-1].strip()
                        current_content = []
                    elif line.startswith('```') and current_file:
                        continue
                    elif current_file:
                        current_content.append(line)
                
                if current_file and current_content:
                    content = '\n'.join(current_content)
                    await self.mcp_fs.write_file(current_file, content)
                    created_files.append(current_file)
            
            await self.mcp_db.log_task("code", task, result[:500])
            
            return AgentResponse(
                success=True,
                result=result,
                files_created=created_files
            )
            
        except Exception as e:
            return AgentResponse(success=False, result=f"代码代理处理失败: {str(e)}")

class DataAgent(BaseAgent):
    """数据分析代理"""
    
    async def process(self, task: str, files: List[str] = [], params: Dict = {}) -> AgentResponse:
        try:
            # 读取数据文件
            data_content = ""
            for file in files:
                if file.endswith(('.csv', '.json', '.txt')):
                    content = await self.mcp_fs.read_file(file)
                    data_content += f"\n文件 {file}:\n{content[:1000]}...\n"
            
            # 获取数据库中的任务历史
            recent_tasks = await self.mcp_db.execute_query(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT 5"
            )
            
            messages = [
                {"role": "system", "content": "你是一个数据分析专家，能够分析各种数据并提供洞察。"},
                {"role": "user", "content": f"""
任务: {task}

数据内容:
{data_content}

最近任务历史:
{json.dumps(recent_tasks, ensure_ascii=False, indent=2)}

请分析数据并提供详细的分析报告和建议。
"""}
            ]
            
            result = await self.deepseek.chat_completion(messages)
            
            # 保存分析报告
            report_file = f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            await self.mcp_fs.write_file(report_file, f"# 数据分析报告\n\n## 任务\n{task}\n\n## 分析结果\n{result}")
            
            await self.mcp_db.log_task("data", task, result[:500])
            
            return AgentResponse(
                success=True,
                result=result,
                files_created=[report_file],
                data={"analysis_summary": result[:200]}
            )
            
        except Exception as e:
            return AgentResponse(success=False, result=f"数据代理处理失败: {str(e)}")

class FileAgent(BaseAgent):
    """文件管理代理"""
    
    async def process(self, task: str, files: List[str] = [], params: Dict = {}) -> AgentResponse:
        try:
            if "列出文件" in task or "list files" in task.lower():
                file_list = await self.mcp_fs.list_files()
                result = f"文件列表:\n{json.dumps(file_list, ensure_ascii=False, indent=2)}"
                
            elif "整理" in task or "organize" in task.lower():
                # 自动整理文件
                file_list = await self.mcp_fs.list_files()
                organized = {"code": [], "data": [], "documents": [], "others": []}
                
                for file_info in file_list:
                    if file_info.get("type") == "file":
                        name = file_info["name"]
                        if name.endswith(('.py', '.js', '.html', '.css')):
                            organized["code"].append(name)
                        elif name.endswith(('.csv', '.json', '.xlsx')):
                            organized["data"].append(name)
                        elif name.endswith(('.md', '.txt', '.pdf')):
                            organized["documents"].append(name)
                        else:
                            organized["others"].append(name)
                
                result = f"文件整理结果:\n{json.dumps(organized, ensure_ascii=False, indent=2)}"
                
            else:
                # 使用AI处理其他文件任务
                file_contents = {}
                for file in files:
                    content = await self.mcp_fs.read_file(file)
                    file_contents[file] = content[:500]  # 限制内容长度
                
                messages = [
                    {"role": "system", "content": "你是一个文件管理助手，能够帮助处理各种文件操作任务。"},
                    {"role": "user", "content": f"""
任务: {task}

相关文件:
{json.dumps(file_contents, ensure_ascii=False, indent=2)}

请执行文件操作任务并说明处理结果。
"""}
                ]
                
                result = await self.deepseek.chat_completion(messages)
            
            await self.mcp_db.log_task("file", task, result[:500])
            
            return AgentResponse(success=True, result=result)
            
        except Exception as e:
            return AgentResponse(success=False, result=f"文件代理处理失败: {str(e)}")

app = FastAPI(title="Multi-Agent MCP System", description="多代理系统")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化组件
deepseek_client = DeepseekClient()
mcp_fs = MCPFileSystem()
mcp_db = MCPDatabase()

# 初始化代理
agents = {
    "code": CodeAgent(deepseek_client, mcp_fs, mcp_db),
    "data": DataAgent(deepseek_client, mcp_fs, mcp_db),
    "file": FileAgent(deepseek_client, mcp_fs, mcp_db)
}

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass

manager = ConnectionManager()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """主页"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>超级无敌工具人</title>
        <meta charset="utf-8">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
            .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
            .header { text-align: center; color: white; margin-bottom: 40px; }
            .header h1 { font-size: 3rem; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
            .header p { font-size: 1.2rem; opacity: 0.9; }
            .card { background: rgba(255,255,255,0.95); border-radius: 15px; padding: 30px; margin-bottom: 30px; box-shadow: 0 8px 32px rgba(0,0,0,0.1); backdrop-filter: blur(10px); }
            .agent-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px; }
            .agent-card { background: white; border-radius: 10px; padding: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); transition: transform 0.3s ease; cursor: pointer; }
            .agent-card:hover { transform: translateY(-5px); }
            .agent-icon { font-size: 2.5rem; margin-bottom: 15px; }
            .control-panel { background: white; border-radius: 10px; padding: 20px; }
            .form-group { margin-bottom: 20px; }
            .form-group label { display: block; margin-bottom: 8px; font-weight: 600; color: #333; }
            .form-control { width: 100%; padding: 12px; border: 2px solid #e1e5e9; border-radius: 8px; font-size: 16px; transition: border-color 0.3s ease; }
            .form-control:focus { outline: none; border-color: #667eea; box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1); }
            .btn { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 12px 30px; border-radius: 8px; font-size: 16px; cursor: pointer; transition: all 0.3s ease; }
            .btn:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.2); }
            .result-area { background: #f8f9fa; border-radius: 8px; padding: 20px; margin-top: 20px; min-height: 200px; border: 2px solid #e1e5e9; font-family: 'Courier New', monospace; white-space: pre-wrap; }
            .status { padding: 10px; border-radius: 5px; margin: 10px 0; }
            .status.success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .status.error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
            .status.info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>超级无敌工具人</h1>
                <p>——————————图灵agent——————————</p>
            </div>
            
            <div class="agent-grid">
                <div class="agent-card" onclick="selectAgent('code')">
                    <div class="agent-icon">💻</div>
                    <h3>代码助手</h3>
                    <p></p>
                </div>
                <div class="agent-card" onclick="selectAgent('data')">
                    <div class="agent-icon">📊</div>
                    <h3>数据分析师</h3>
                    <p></p>
                </div>
                <div class="agent-card" onclick="selectAgent('file')">
                    <div class="agent-icon">📁</div>
                    <h3>文件管理器</h3>
                    <p></p>
                </div>
            </div>
            
            <div class="card">
                <div class="control-panel">
                    <div class="form-group">
                        <label for="agent-select">选择代理:</label>
                        <select id="agent-select" class="form-control">
                            <option value="code">💻 代码助手</option>
                            <option value="data">📊 数据分析师</option>
                            <option value="file">📁 文件管理器</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="task-input">任务描述:</label>
                        <textarea id="task-input" class="form-control" rows="3" placeholder="请描述您需要完成的任务..."></textarea>
                    </div>
                    
                    <div class="form-group">
                        <label for="files-input">相关文件 (可选):</label>
                        <input type="text" id="files-input" class="form-control" placeholder="文件路径，用逗号分隔">
                    </div>
                    
                    <button onclick="executeTask()" class="btn"> 执行任务</button>
                    <button onclick="listFiles()" class="btn" style="margin-left: 10px;"> 列出文件</button>
                </div>
                
                <div id="result" class="result-area">等待任务执行...</div>
            </div>
        </div>
        
        <script>
            function selectAgent(agentType) {
                document.getElementById('agent-select').value = agentType;
                const examples = {
                    'code': '生成一个Python爬虫脚本',
                    'data': '分析CSV文件中的销售数据',
                    'file': '整理文件夹中的文档'
                };
                document.getElementById('task-input').placeholder = examples[agentType] || '请描述您的任务...';
            }
            
            async function executeTask() {
                const agentType = document.getElementById('agent-select').value;
                const task = document.getElementById('task-input').value;
                const files = document.getElementById('files-input').value.split(',').map(f => f.trim()).filter(f => f);
                
                if (!task) {
                    alert('请输入任务描述');
                    return;
                }
                
                const resultDiv = document.getElementById('result');
                resultDiv.innerHTML = '<div class="status info">正在处理任务...</div>';
                
                try {
                    const response = await fetch('/api/execute', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            agent_type: agentType,
                            task: task,
                            files: files
                        })
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        let output = `<div class="status success">✅ 任务执行成功</div>`;
                        output += `<strong>结果:</strong>\n${result.result}\n\n`;
                        if (result.files_created.length > 0) {
                            output += `<strong>创建的文件:</strong>\n${result.files_created.join(', ')}\n\n`;
                        }
                        if (result.data) {
                            output += `<strong>附加数据:</strong>\n${JSON.stringify(result.data, null, 2)}`;
                        }
                        resultDiv.innerHTML = output;
                    } else {
                        resultDiv.innerHTML = `<div class="status error">❌ 任务执行失败</div>${result.result}`;
                    }
                } catch (error) {
                    resultDiv.innerHTML = `<div class="status error">❌ 请求失败: ${error.message}</div>`;
                }
            }
            
            async function listFiles() {
                const resultDiv = document.getElementById('result');
                resultDiv.innerHTML = '<div class="status info">正在获取文件列表...</div>';
                
                try {
                    const response = await fetch('/api/files');
                    const files = await response.json();
                    
                    let output = '<div class="status success">📁 文件列表</div>';
                    output += JSON.stringify(files, null, 2);
                    resultDiv.innerHTML = output;
                } catch (error) {
                    resultDiv.innerHTML = `<div class="status error">❌ 获取文件列表失败: ${error.message}</div>`;
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/api/execute")
async def execute_task(request: TaskRequest):
    """执行任务"""
    if request.agent_type not in agents:
        raise HTTPException(status_code=400, detail="不支持的代理类型")
    
    agent = agents[request.agent_type]
    result = await agent.process(request.task, request.files, request.params)
    
    return result

@app.get("/api/files")
async def list_files():
    """获取文件列表"""
    return await mcp_fs.list_files()

@app.get("/api/tasks")
async def get_tasks():
    """获取任务历史"""
    tasks = await mcp_db.execute_query("SELECT * FROM tasks ORDER BY created_at DESC LIMIT 20")
    return tasks

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket连接"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.broadcast(f"收到消息: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    print("—" * 60)
    print(" 图灵机器人启动中...")
    print("=" * 60)
    print("上课专用")
    print("访问地址: http://localhost:8000")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")