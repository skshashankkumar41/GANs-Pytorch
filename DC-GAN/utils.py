import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

def noise_generator(n_samples,z_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noise = torch.randn(n_samples,z_dim).to(device)
    return noise

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28),tensorboard_writer=False):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in a uniform grid.
    '''
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)

    if tensorboard_writer:
        return image_grid
    else:
        plt.imshow(image_grid.permute(1, 2, 0).squeeze())
        plt.show()
        return None

def tensorboard_writer(img_grid,epoch,step,gen_image=True,path='logs/basic_gan'):
    img_name = 'gen_epoch_{}_step_{}'.format(epoch,step) if gen_image else 'real_epoch_{}_step_{}'.format(epoch,step)
    writer = SummaryWriter(path)
    writer.add_image(img_name, img_grid)
    return None