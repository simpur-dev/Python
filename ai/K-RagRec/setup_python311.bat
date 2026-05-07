@echo off
REM 设置 Python 3.11 为默认 Python
REM 临时修改 PATH，优先使用 Python 3.11

echo ============================================================
echo 设置 Python 3.11 环境
echo ============================================================

REM 将 Python 3.11 添加到 PATH 最前面
set "PYTHON311_PATH=C:\Users\34928\AppData\Local\Programs\Python\Python311"
set "PATH=%PYTHON311_PATH%;%PYTHON311_PATH%\Scripts;%PATH%"

echo.
echo Python 版本检查:
python --version

echo.
echo PyTorch 检查:
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

echo.
echo ============================================================
echo Python 3.11 环境已设置
echo ============================================================
echo.
echo 现在可以运行:
echo   python run_full_test.py
echo   python main.py --mode demo --use_sample_data
echo.

REM 保持窗口打开，进入交互模式
cmd /k
