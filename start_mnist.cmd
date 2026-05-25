@echo off
call "%USERPROFILE%\miniconda3\condabin\activate.bat" mnist-nn
echo.
echo mnist-nn environment is active.
echo Python:
python --version
echo.
cmd /k
