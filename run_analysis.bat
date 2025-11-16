@echo off
REM Windows Batch Script for running the Stock Analysis Pipeline
REM Place this file in project root and schedule it with Task Scheduler

setlocal enabledelayedexpansion

REM Get current directory
set PROJECT_DIR=%~dp0
cd /d %PROJECT_DIR%

REM Setup logging
if not exist "logs" mkdir logs
set LOG_FILE=logs\batch_run_%date:~10,4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%.log
set LOG_FILE=%LOG_FILE: =0%

REM Log start
echo [%date% %time%] Starting Stock Analysis Pipeline >> "%LOG_FILE%"

REM Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    echo [%date% %time%] Activating virtual environment >> "%LOG_FILE%"
    call venv\Scripts\activate.bat
) else (
    echo [%date% %time%] ERROR: Virtual environment not found >> "%LOG_FILE%"
    echo Virtual environment not found. Please run setup first.
    exit /b 1
)

REM Run the pipeline
echo [%date% %time%] Running main.py >> "%LOG_FILE%"
python main.py >> "%LOG_FILE%" 2>&1
set PYTHON_EXIT=%errorlevel%

REM Check result
if %PYTHON_EXIT% equ 0 (
    echo [%date% %time%] Pipeline completed successfully >> "%LOG_FILE%"
    echo SUCCESS: Pipeline ran successfully. Check logs for details.
) else (
    echo [%date% %time%] ERROR: Pipeline failed with exit code %PYTHON_EXIT% >> "%LOG_FILE%"
    echo ERROR: Pipeline failed. Check %LOG_FILE% for details.
)

echo [%date% %time%] Finished >> "%LOG_FILE%"

endlocal
exit /b %PYTHON_EXIT%
