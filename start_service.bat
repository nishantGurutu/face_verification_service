@echo off
cd /d D:\my-space\work-book-space\face_verify_service
.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
pause
