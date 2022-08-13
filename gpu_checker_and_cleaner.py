import torch

k = torch.cuda.is_available()

print(k)
print(torch.__version__)
print(torch.version.cuda)
torch.cuda.empty_cache()