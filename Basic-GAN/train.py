from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch.nn as nn
from generator import Generator
from discriminator import Discriminator
from utils import noise_generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloader(batch_size):
    dataloader = DataLoader(MNIST('.',transform=transforms.ToTensor(),batch_size=batch_size,shuffle=True))
    return dataloader

def discriminator_loss(generator,discriminator,criterion,real,num_images,z_dim):
    noise = noise_generator(num_images,z_dim)
    gen_out = generator(noise)
    # discriminator output on fake images generated by generator
    disc_fake_out = discriminator(gen_out.detach())
    # calculating loss on fake images as discrimiator wants theses images to be fake(0)
    disc_fake_loss = criterion(disc_fake_out,torch.zeros_like(disc_fake_out))

    #discriminator output on real images
    disc_real_out = discriminator(real)
    # calculating loss on real images as discrimiator wants theses images to be real(1) 
    disc_real_loss = criterion(disc_real_out,torch.ones_like(disc_real_out))

    #averaging both loss
    disc_loss = (disc_real_loss+disc_fake_loss)/2

    return disc_loss


def generator_loss(generator,discriminator,criterion,num_images,z_dim):
    noise = noise_generator(num_images,z_dim)
    gen_out = generator(noise)
    # discriminator output on fake images generated by generator
    gen_fake_out = discriminator(gen_out.detach())
    # calculating loss on fake images as generator wants discrimiator to treat theses images as real(1)
    gen_fake_loss = criterion(gen_fake_out,torch.ones_like(gen_fake_out))
    
    return gen_fake_loss

def train(n_epochs=200,batch_size=128,lr=0.00001,z_dim=64):
    criterion = nn.BCEWithLogitsLoss()
    # generator model with optimizer
    generator = Generator(z_dim).to(device)
    generator_opt = torch.optim.Adam(generator.parameters(),lr=lr)

    # discriminator model with optimizer
    discriminator = Discriminator().to_device()
    discriminator_opt = torch.optim.Adam(discriminator.parameters(),lr=lr)

    

