@echo off
setlocal
cd /d "%~dp0"
echo Starting Use Mode...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_use.ps1"
if errorlevel 1 (
echo.
echo run_use failed. Press any key to close.
pause >nul
)
