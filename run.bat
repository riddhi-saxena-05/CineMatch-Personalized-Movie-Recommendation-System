@echo off
cd /d "%~dp0"
if exist .venv\Scripts\python.exe (
    start http://127.0.0.1:5000
    .venv\Scripts\python.exe app.py
) else (
    echo Virtual environment not found. Please run: uv venv .venv ^&^& uv pip install -r requirements.txt
    pause
)
