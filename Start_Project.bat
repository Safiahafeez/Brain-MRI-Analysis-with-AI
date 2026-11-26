@echo off
cd /d "C:\Users\Ahmad\Downloads\brain-tumor-ai"
start "FastAPI Backend" cmd /k "uvicorn backend:app"
start "Streamlit Frontend" cmd /k "streamlit run frontend.py"
start http://localhost:8501
exit