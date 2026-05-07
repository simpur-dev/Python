import torch

print("=" * 60)
print("本地 GPU 检测报告")
print("=" * 60)

print(f"\n[1] PyTorch 版本: {torch.__version__}")
print(f"[2] CUDA 编译版本: {torch.version.cuda}")

print(f"\n[3] CUDA 是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"[4] 当前 CUDA 设备索引: {torch.cuda.current_device()}")
    print(f"[5] GPU 设备名称: {torch.cuda.get_device_name(0)}")
    
    props = torch.cuda.get_device_properties(0)
    print(f"[6] GPU 显存总量: {round(props.total_memory / 1024**3, 2)} GB")
    print(f"[7] GPU 计算能力: {props.major}.{props.minor}")
    print(f"[8] GPU 处理器数量: {props.multi_processor_count}")
    
    print("\n[9] 实际运行测试:")
    x = torch.randn(5000, 5000, device='cuda')
    y = torch.randn(5000, 5000, device='cuda')
    
    torch.cuda.synchronize()
    import time
    start = time.time()
    for _ in range(10):
        z = torch.matmul(x, y)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"    - 张量 x 设备: {x.device}")
    print(f"    - 张量 y 设备: {y.device}")
    print(f"    - 矩阵乘法(5000x5000) x 10 次耗时: {elapsed:.3f} 秒")
    
    print("\n" + "=" * 60)
    print("结论: 正在使用本地 GPU (NVIDIA RTX 4070 Laptop)")
    print("=" * 60)
else:
    print("\n警告: CUDA 不可用，未检测到本地 GPU")
