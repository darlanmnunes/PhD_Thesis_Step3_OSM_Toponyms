import torch

print("Versão do PyTorch:", torch.__version__)
print("CUDA disponível:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Nome da GPU:", torch.cuda.get_device_name(0))
    print("Versão do CUDA:", torch.version.cuda)
else:
    print("Nenhuma GPU CUDA disponível.")