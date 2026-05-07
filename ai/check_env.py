import torch
print(f"PyTorch: {torch.__version__}")

try:
    import transformers
    print(f"Transformers: {transformers.__version__}")
except ImportError:
    print("Transformers: Not installed")

try:
    import sentence_transformers
    print(f"SentenceTransformers: {sentence_transformers.__version__}")
except ImportError:
    print("SentenceTransformers: Not installed")

try:
    import torch_geometric
    print(f"PyG: {torch_geometric.__version__}")
except ImportError:
    print("PyG: Not installed")

try:
    import faiss
    print(f"FAISS: installed")
except ImportError:
    print("FAISS: Not installed")

print(f"\nCUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
