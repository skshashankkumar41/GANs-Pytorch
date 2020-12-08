import torch 
import torch.nn as nn 

# discriminator class to take noise vector and return the image
class Discriminator(nn.Module):
    def __init__(self,im_dim,hidden_dim):
        super(Discriminator,self).__init__()
        self.disc = nn.Sequential(

        )

    def discriminator_block(self,input_dim,output_dim):
        
