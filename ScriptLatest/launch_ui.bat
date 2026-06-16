@echo off
echo.
echo  ================================================
echo   A/B Experiment Runner - Streamlit UI Launcher
echo  ================================================
echo.
cd /d "%~dp0"
echo Starting at http://localhost:8501
echo.
python -m streamlit run ui_app.py --server.port 8501 --browser.gatherUsageStats false
pause
