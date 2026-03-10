@echo off
setlocal
cd /d "%~dp0"
echo Resetting dev environment...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\reset_dev.ps1"
if errorlevel 1 (
echo.
echo reset_dev failed. Press any key to close.
pause >nul
)
