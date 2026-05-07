@echo off
REM K-RagRec 快速启动脚本 (使用 Python 3.11)

echo ============================================================
echo K-RagRec 快速启动 (Python 3.11)
echo ACL 2025
echo ============================================================

REM 设置 Python 3.11 路径
set "PYTHON311=C:\Users\34928\AppData\Local\Programs\Python\Python311\python.exe"

REM 步骤 1: 环境检查
echo.
echo [步骤 1/5] 检查环境...
%PYTHON311% ..\check_env.py
if %errorlevel% neq 0 (
    echo [错误] 环境检查失败
    pause
    exit /b 1
)
echo [成功] 环境检查通过

REM 步骤 2: 下载数据
echo.
echo [步骤 2/5] 检查数据集...
if exist "data\ml-1m" (
    echo [成功] 数据集已存在，跳过下载
) else (
    echo [提示] 数据集不存在，运行下载脚本...
    %PYTHON311% download_data.py
    if %errorlevel% neq 0 (
        echo [警告] 数据下载失败，将使用示例数据
    ) else (
        echo [成功] 数据下载完成
    )
)

REM 步骤 3: 模块测试
echo.
echo [步骤 3/5] 运行模块测试...
%PYTHON311% run_full_test.py
if %errorlevel% neq 0 (
    echo [错误] 模块测试失败
    pause
    exit /b 1
)
echo [成功] 模块测试通过

REM 步骤 4: Demo 运行
echo.
echo [步骤 4/5] 运行 Demo...
%PYTHON311% main.py --mode demo --use_sample_data
if %errorlevel% neq 0 (
    echo [错误] Demo 运行失败
    pause
    exit /b 1
)
echo [成功] Demo 运行成功

REM 步骤 5: 完成
echo.
echo ============================================================
echo 快速启动完成!
echo ============================================================

echo.
echo 下一步操作:
echo   1. 训练模型: %PYTHON311% main.py --mode train
echo   2. 评估模型: %PYTHON311% main.py --mode eval
echo   3. Ablation Study: %PYTHON311% run_ablation_study.py
echo   4. 完整实验: %PYTHON311% run_experiments.py

echo.
echo 查看详细文档: ..\REPRODUCTION_STEPS.md

pause
