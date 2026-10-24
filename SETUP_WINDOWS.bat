@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  py -3.11 -m venv .venv
  if errorlevel 1 goto fail
)
.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto fail
echo Setup complete. In VS Code terminal run: .\.venv\Scripts\python.exe main.py
pause
exit /b
:fail
echo Setup failed. Read the actual error above. Close other installers before retrying.
pause
