@echo off
chcp 65001 >nul
echo ========================================
echo   Install remaining Python packages
echo ========================================
echo.

call "d:\BaiduNetdiskDownload\AI大模型Agent智能体开发篇\.venv\Scripts\activate.bat"

echo [1/3] Installing LlamaIndex packages...
pip install llama-index llama-index-llms-dashscope llama-index-embeddings-huggingface llama-index-vector-stores-chroma

echo.
echo [2/3] Installing vector DB and tools...
pip install chromadb faiss-cpu

echo.
echo [3/3] Installing doc processing and frontend...
pip install gradio pypdf python-docx markdown numpy pandas

echo.
echo ========================================
echo   All packages installed!
echo ========================================
echo.
echo To activate venv, run:
echo   d:\BaiduNetdiskDownload\AI大模型Agent智能体开发篇\.venv\Scripts\activate.bat
pause
