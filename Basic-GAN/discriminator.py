import torch 
import torch.nn as nn 

# discriminator class to take noise vector and return the image
class Discriminator(nn.Module):
    def __init__(self,im_dim,hidden_dim):
        super(Discriminator,self).__init__()
        self.disc = nn.Sequential(
            self.discriminator_block(im_dim,hidden_dim*4),
            self.discriminator_block(hidden_dim*4,hidden_dim*2),
            self.discriminator_block(hidden_dim*2,hidden_dim),
            nn.Linear(hidden_dim,1)
        )
    # discriminator block 
    def discriminator_block(self,input_dim,output_dim):
        return nn.Sequential(
            nn.Linear(input_dim,output_dim),
            nn.LeakyReLU(0.2)
        )

    def forward(self,image):
        return self.disc(image)
