import torch 
import torch.nn as nn 

# generator class to take noise vector and return the image
class Generator(nn.Module):
    def __init__(self,z_dim=10,img_chan=1,hidden_dim=64):
        super(Generator,self).__init__()
        self.z_dim = z_dim
        self.gen = nn.Sequential(
            self.generator_block(z_dim, hidden_dim*4),
            self.generator_block(hidden_dim*4, hidden_dim*2,kernel_size=4,stride=1),
            self.generator_block(hidden_dim*2, hidden_dim),
            self.generator_block(hidden_dim, img_chan, kernel_size=4, final_layer=True),

        )

    # block of generator 
    def generator_block(self,input_channel,output_channel,kernel_size=3,stride=2,final_layer=False):
        if not final_layer:
            nn.Sequential(
                nn.ConvTranspose2d(input_channel,output_channel,kernel_size,stride),
                nn.BatchNorm2d(output_channel),
                nn.ReLU()
            )
        else:
            nn.Sequential(
                nn.ConvTranspose2d(input_channel,output_channel,kernel_size,stride),
                nn.Tanh()
            )

    def unsqueeze_noise(self,noise):
        return noise.view(len(noise), self.z_dim, 1, 1)

    def forward(self,noise):
        x = self.unsqueeze_noise(noise)
        return self.gen(x)
