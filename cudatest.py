import torch
print(torch.cuda.is_available())  # 检查 CUDA 是否可用 # check if CUDA is available
print(torch.cuda.device_count())  # 检查可用 GPU 数量 # check available GPU count

# 检查可用 GPU 数量
# Check available GPU count
available_gpus = torch.cuda.device_count()
if available_gpus == 0:
    raise RuntimeError("No GPUs are available. Please check your CUDA setup.")

# 动态设置 GPU 列表
# Dynamically set GPU list
gpu = list(range(available_gpus))
print(gpu)

# 使用 DataParallel
# Use DataParallel
# encoder = torch.nn.DataParallel(encoder, device_ids=gpu)

# 检查 CUDA 是否可用
# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 示例模型和数据
# Example model and data
model = torch.nn.Linear(10, 1).to(device)  # 将模型加载到 GPU # Load model to GPU
data = torch.randn(5, 10).to(device)       # 将数据加载到 GPU # Load data to GPU

# 前向传播
# Forward pass
output = model(data)
print(output)