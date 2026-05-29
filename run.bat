@echo off
cd /d "%~dp0"
if "%~1"=="" (
  echo Usage:
  echo   run.bat setup       Install dependencies and models
  echo   run.bat demo        Run pipeline demo
  echo   run.bat hand        Run hand sensor webcam
  echo   run.bat inference   Run webcam gesture control
  exit /b 1
)
python run.py %*
