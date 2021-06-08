import torch

# helper function to get a device
# if cpu = True, just use the CPU as a device.
# else, try to get the GPU
def get_device(cpu):
    if cpu or not torch.cuda.is_available(): return torch.device('cpu')
    return torch.device('cuda')

# given a PyTorch module, calculate the number of trainable parameters
def get_parameter_count(net: torch.nn.Module) -> int:
    return sum(p.numel() for p in net.parameters() if p.requires_grad)
