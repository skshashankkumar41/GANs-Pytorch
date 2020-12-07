import torch 
import torch.nn as nn 

# block of generator 
def generator_block(input_dim,output_dim):
    return nn.Sequential(
        nn.Linear(input_dim,output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU(inplace=True)
    )

# generator class to take noise vector and return the image
class Generator(nn.Module):
    def __init__(self,z_dim,img_dim,hidden_dim):
        super(Generator,self).__init__()
        self.gen = nn.Sequential(
            generator_block(z_dim, hidden_dim),
            generator_block(hidden_dim, hidden_dim*2),
            generator_block(hidden_dim*2, hidden_dim*4),
            generator_block(hidden_dim*4, hidden_dim*8),
            nn.Linear(hidden_dim*8, img_dim),
            nn.Sigmoid()
        )

    def forward(self,noise):
        return self.gen(noise)