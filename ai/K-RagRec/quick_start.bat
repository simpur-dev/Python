@echo off
REM K-RagRec 快速启动脚本 (Windows)
REM 一键完成环境检查、数据下载、模块测试

echo ============================================================
echo K-RagRec 快速启动
echo ACL 2025
echo ============================================================

REM 步骤 1: 环境检查
echo.
echo [步骤 1/5] 检查环境...
python ..\check_env.py
if %errorlevel% neq 0 (
    echo [错误] 环境检查失败，请先安装依赖
    pause
    exit /b 1
)
echo [成功] 环境检查通过

REM 步骤 2: 下载数据
echo.
echo [步骤 2/5] 下载数据集...
if exist "K-RagRec\data\ml-1m" (
    echo [成功] 数据集已存在，跳过下载
) else (
    python K-RagRec\download_data.py
    if %errorlevel% neq 0 (
        echo [错误] 数据下载失败
        pause
        exit /b 1
    )
    echo [成功] 数据下载完成
)

REM 步骤 3: 模块测试
echo.
echo [步骤 3/5] 运行模块测试...
python K-RagRec\run_full_test.py
if %errorlevel% neq 0 (
    echo [错误] 模块测试失败
    pause
    exit /b 1
)
echo [成功] 模块测试通过

REM 步骤 4: Demo 运行
echo.
echo [步骤 4/5] 运行 Demo...
python K-RagRec\main.py --mode demo --use_sample_data
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
echo   1. 训练模型: python K-RagRec\main.py --mode train
echo   2. 评估模型: python K-RagRec\main.py --mode eval
echo   3. Ablation Study: python K-RagRec\run_ablation_study.py
echo   4. 完整实验: python K-RagRec\run_experiments.py

echo.
echo 查看详细文档: REPRODUCTION_STEPS.md

pause
