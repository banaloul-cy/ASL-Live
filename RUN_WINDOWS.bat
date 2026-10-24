@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Please run SETUP_WINDOWS.bat first.
  pause
  exit /b 1
)
.venv\Scripts\python.exe main.py
if errorlevel 1 pause
