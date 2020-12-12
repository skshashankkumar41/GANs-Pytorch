import torch

def noise_generator(n_samples,z_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noise = torch.randn(n_samples,z_dim).to(device)
    return noise