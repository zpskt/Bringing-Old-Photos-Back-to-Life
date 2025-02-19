import torch
if __name__ == '__main__':
    print(torch.__version__)
    print(torch.version.cuda)
    # 如果你是苹果M芯片

    # 检查是否支持 Metal
    if torch.backends.mps.is_available():
        print("MPS is available!")
        device = torch.device("mps")
    else:
        print("MPS is not available.")
        device = torch.device("cpu")
    # 创建 tensor
    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    print(x)