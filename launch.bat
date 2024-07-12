@echo off
echo Launching TagScribeR...

:: Activate the virtual environment
call venv\Scripts\activate.bat

:: Run the main Python script
python main.py

:: Deactivate the virtual environment when the program exits
deactivate

pause