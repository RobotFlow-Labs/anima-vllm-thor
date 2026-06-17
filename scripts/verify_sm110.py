import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda)
print("capability", torch.cuda.get_device_capability(0), "|", torch.cuda.get_device_name(0))
x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
y = (x @ x).sum().item()
print("MATMUL_OK sum=", round(y, 2))
