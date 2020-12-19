import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def noise_generator(n_samples,z_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noise = torch.randn(n_samples,z_dim).to(device)
    return noise

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in a uniform grid.
    '''
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()