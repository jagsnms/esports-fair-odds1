Set-Location (Join-Path $PSScriptRoot "..")
.\.venv311\Scripts\python -m uvicorn backend.main:app --reload --port 8000 --reload-dir backend --reload-dir engine
