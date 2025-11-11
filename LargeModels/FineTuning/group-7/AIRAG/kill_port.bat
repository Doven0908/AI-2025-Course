@echo off
REM Kill processes occupying ports 7860-7864

echo Checking ports 7860-7864...
echo.

for /L %%i in (7860,1,7864) do (
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%%i') do (
        echo Killing process %%a on port %%i
        taskkill /F /PID %%a 2>nul
    )
)

echo.
echo Done!
pause



