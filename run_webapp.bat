@echo off
REM Batch script to run the webapp with proper environment

cd /d "%~dp0"

REM Add Rust DLL path to system PATH
set PATH=%CD%\src\aion26_rust\target\release;%PATH%

REM Run with venv Python
.venv\Scripts\python.exe scripts\train_webapp_pro.py

pause
