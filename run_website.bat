@echo off
title MMAI Traffic Analysis & Accident Severity AI System
cd /d "%~dp0"

echo ===================================================================
echo  🚗 MMAI - Automated Traffic Analysis, Accident Detection,
echo     Severity Assessment, and Vehicle Counting System
echo     Powered by YOLOv11 + ByteTrack + Gradio
echo ===================================================================
echo  Starting Web Application on http://localhost:7860 ...
echo  Press Ctrl+C to stop the server at any time.
echo ===================================================================

start "" "http://localhost:7860"
python app.py
pause

