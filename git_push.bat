@echo off
chcp 65001 >nul
echo Pushing to GitHub...
cd /d "d:\BaiduNetdiskDownload\AI大模型Agent智能体开发篇"
git push -u origin main
if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo   Push SUCCESS!
    echo   https://github.com/799193396/agent-example
    echo ========================================
) else (
    echo.
    echo Push FAILED. Check SSH key and network.
)
pause
