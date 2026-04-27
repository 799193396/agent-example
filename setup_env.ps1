# Python Environment Setup Script
$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvDir = Join-Path $projectDir ".venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"
$venvPip = Join-Path $venvDir "Scripts\pip.exe"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Python Environment Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if (Test-Path $venvPython) {
    Write-Host "[OK] venv exists: $venvDir" -ForegroundColor Green
} else {
    Write-Host "[...] Creating venv..." -ForegroundColor Yellow
    python -m venv $venvDir
    Write-Host "[OK] venv created" -ForegroundColor Green
}

$pyVersion = & $venvPython --version
Write-Host "[INFO] Python: $pyVersion" -ForegroundColor Cyan

Write-Host "[...] Upgrading pip..." -ForegroundColor Yellow
& $venvPython -m pip install --upgrade pip | Out-Null
Write-Host "[OK] pip upgraded" -ForegroundColor Green

Write-Host "`n[1/4] Installing Agent core deps (langchain)..." -ForegroundColor Yellow
& $venvPip install langchain langchain-openai langchain-tavily langchainhub python-dotenv

Write-Host "`n[2/4] Installing LlamaIndex core deps..." -ForegroundColor Yellow
& $venvPip install llama-index llama-index-llms-dashscope llama-index-embeddings-huggingface llama-index-vector-stores-chroma

Write-Host "`n[3/4] Installing vector DB and search tools..." -ForegroundColor Yellow
& $venvPip install chromadb faiss-cpu tiktoken

Write-Host "`n[4/4] Installing doc processing and frontend..." -ForegroundColor Yellow
& $venvPip install gradio pypdf python-docx markdown openai numpy pandas

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Activate venv:" -ForegroundColor Yellow
Write-Host "  $venvDir\Scripts\Activate.ps1" -ForegroundColor White
