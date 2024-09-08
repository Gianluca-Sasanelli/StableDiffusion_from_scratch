import torch,math
import torch.nn as nn
import torch.nn.functional as F
# Main ideas behind the paper/model https://arxiv.org/pdf/2112.10752:
# 1)Doing the diffusion process on a lower dimensional space (obtained 
# with an autoencoder). It is computationally more efficient than doing it on pixel levels.
# 2) I don't understand it now
# 3)Obtaining "general-purpose" compression models whose latent space
# can be used to train multiple generative models.

# The autoencoder is trained with a perceptual loss and a patch-based adversarial objective.
""""

What is a diffusion model https://arxiv.org/pdf/2006.11239? https://lilianweng.github.io/posts/2021-07-11-diffusion-models/

Probabilistic model that learn the data distribution p(x) 
by denoising a normally distributed variable. The denoising consists in  markov chain.
The loss would be an MSE between the expected noise by the model and the real noise.
The neural backbone that predicts the noise is a UNET.


"""

#Let's start with a UNET
#The idea is to have (128)^2 pixel images, with time embeddings
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels,time_steps: int = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels , stride= 1, padding = 1,kernel_size = 3) 
        self.norm1 = nn.BatchNorm2d(out_channels) 
        self.act1 = nn.SiLU() 
        
        self.conv2 = nn.Conv2d(out_channels, out_channels , stride= 1, padding = 1,kernel_size = 3) 
        self.norm2 = nn.BatchNorm2d(out_channels) 
        self.act2 = nn.SiLU() 

        #half of the size
        self.down = nn.Sequential(
            nn.Conv2d(out_channels, out_channels , stride= 2, padding = 1,kernel_size = 3),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
            )
        
        if time_steps is not None:
            self.time_emb = nn.Embedding(time_steps, out_channels)
        if in_channels == out_channels:
            self.conv_shortcut = nn.Identity()
        else:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding = 1)
        
    def forward(self, input: torch.Tensor,t: torch.Tensor = None):
        x = self.act1(self.norm1(self.conv1(input)))
        #adding the time embedding between the two convolutions
        x += self.time_emb(t)[:,:,None, None]
        res = self.act2(self.norm2(self.conv2(x)))
        res += self.conv_shortcut(input)
        x= self.down(res)
        return x, res
    
class UpSample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        assert in_channels % 2 == 0, "in_channels must be even"
        self.horizontal = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2 , stride= 1, padding = 1,kernel_size = 3),
            nn.BatchNorm2d(in_channels // 2),
            nn.SiLU(),
            nn.Conv2d(in_channels // 2, in_channels // 4, stride= 1, padding = 1,kernel_size = 3),
            nn.BatchNorm2d(in_channels // 4),
            nn.SiLU(),
            )
        #half of the size
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4 , stride= 2, padding = 1,kernel_size = 3, output_padding= 1),
            nn.BatchNorm2d(in_channels // 4),
            nn.SiLU()
            )
        
    def forward(self, x: torch.Tensor):
        x = self.horizontal(x)
        x = self.up(x)
        return x
        
class UNET(nn.Module):
    #supposing that the image size is 128
    def __init__(self, in_channels: int, latent_dim: int = 128, time_steps:int = 50, hidden_dims = None):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.time_steps = time_steps
        #Dimensionality of the feature channels
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512, 1024]
        hidden_dims.insert(0, in_channels) 
        
        self.final_hidden_dim = hidden_dims[-1]
        #Adding the downsample, horizontal convolutions with residuals and time embeddings
        self.downpass = nn.ModuleList(
            [DownSample(in_channels = hidden_dims[i], out_channels = hidden_dims[i+1], time_steps= time_steps) for i in range(len(hidden_dims) -1 )]) 
        #encoder-decoder in the latent dimension
        self.enc_lin = nn.Linear(self.final_hidden_dim*4, latent_dim)
        self.dec_lin = nn.Linear(latent_dim, self.final_hidden_dim * 16)
        
        hidden_dims.reverse()
        #Symmetrical opposit of the downpass, without the time embeddings
        self.uppass = nn.ModuleList(
            [UpSample(in_channels = hidden_dims[i]* 2) for i in range(len(hidden_dims) - 2 )])
        self.final_layer = nn.Sequential( 
                                         nn.Conv2d(in_channels= hidden_dims[-2] , out_channels=hidden_dims[-1], kernel_size=3,
                                                   stride=1, padding=1),
                                         nn.Tanh()
                                         )
            
        hidden_dims.reverse()
#from image to latent dimension        
    def encoder(self, x: torch.Tensor, t:torch.Tensor = None):
        residuals = []
        for down in self.downpass:
            x, res = down(x, t)
            residuals.append(res)
        x = torch.flatten(x, start_dim= 1)
        x = self.enc_lin(x)
        return x, residuals
#from latent dimension to image
    def decoder(self, x: torch.Tensor, residuals: list):
        x = self.dec_lin(x)
        x = x.view(-1, self.final_hidden_dim, 4, 4)
        for i, up in enumerate(self.uppass):
            x = torch.cat((x, residuals[i]), dim = 1)
            x = up(x)
        return x
    
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x, residuals = self.encoder(x, t)
        residuals.reverse()
        x = self.decoder(x, residuals)
        x = self.final_layer(x)
        return x

"""" Let's try to reproduce the DDPM paper: https://arxiv.org/pdf/2006.11239.
     Basically, we add noise to the data distribution q(xt|xt-1) and then we learn the distribution p(xt-1|xt).
     We cannot just reverse q(xt-1|xt) because it is intractable. 
     However, we can reverse q(xt-1| xt,x0), the reverse conditioned on the previous step and x0.
     Basically, this reverse can be written as function of the reconstructed noise.
     Now, defining the ELBO loss to minimize the difference between q reverse and p, which in practise is an MSE loss between the added noise
     at time step t and the reconstructed noise at the same time step.
     
     The model that gives us the eps_theta should work on images such as a Unet."""
class DiffusionModel(nn.Module):
    def __init__(self, in_channels:int = 3, latent_dim:int = 256, hidden_dims = None, beta_min: float= 0.0001, beta_max: float = 0.02, steps:int = 50, device = "cuda" ):
        super().__init__()
        #The model that returns the noise which is with the same shape of the image
        self.reconstructed_noise = UNET(in_channels = in_channels, latent_dim= latent_dim, time_steps= steps, hidden_dims=hidden_dims)
        #time steps
        self.steps = steps
        #variance scheduler
        self.beta = torch.linspace(beta_min, beta_max, steps).to(device)
        #necessary to compute the mean of distributions
        self.alpha = (1 - self.beta).to(device)
        self.alpha_tot = torch.cumprod(self.alpha, dim = 0).to(device)
    """Adding noise to the distribution, sampling at any t with the reparametrization trick
    q(xt|x0) = N(xt; alpha_tot^0.5 * x0, (1-alpha_tot) * I)
    """
    def sample_q(self, x0: torch.Tensor, t: torch.Tensor, eps= None):
        mean = torch.sqrt(self.alpha_tot[t].view(-1,1,1,1)) * x0  
        var =  1 - self.alpha_tot[t].view(-1,1,1,1)
        if eps is None: 
            eps = torch.randn_like(x0)
        sample = mean + torch.sqrt(var) * eps        
        return sample, eps
    """denoising the distribution at time step t
    p(xt-1|xt ) = N(xt-1; mu(xt,t), var(xt, t))
    """
    def sample_p(self, xt: torch.Tensor, t: torch.Tensor):
        eps_theta = self.reconstructed_noise(xt, t)
        alpha_tot = self.alpha_tot[t].view(-1,1,1,1)
        alpha = self.alpha[t].view(-1,1,1,1)
        coef = (1 - alpha) / torch.sqrt(1- alpha_tot)
        mean = (1 / torch.sqrt(alpha)) * (xt - coef * eps_theta)
        var = self.beta[t].view(-1,1,1,1)
        eps = torch.randn(xt.shape, device = xt.device)
        sample = mean + (torch.sqrt(var)) *eps
        return sample, eps_theta
    
    
    """ The loss is between the added noise and the reconstructed noise """
    
#Computing the loss between the real noise and epsilon    
    def loss(self, x0:torch.Tensor, noise = None):
        batch_size = x0.shape[0]
        t = torch.randint(low = 0, high = self.steps, size= (batch_size,), device = x0.device, dtype = torch.long )
        if noise is None:
            noise = torch.randn_like(x0)
        xt, _ = self.sample_q(x0, t, eps = noise)
        eps_theta = self.reconstructed_noise(xt, t)
        return F.mse_loss(noise, eps_theta)
    
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @torch.no_grad()
    def generate_image(self, img_shape, starting_image = None):
        """
        Generate an image starting from total noise by reversing the diffusion process.
        
        Args:
            img_shape (tuple): Shape of the image to generate (e.g., (batch_size, channels, height, width)).
        
        Returns:
            torch.Tensor: Generated images.
        """
        # Step 1: Initialize pure noise (final step in the forward process)
        xt = torch.randn(img_shape, device=self.alpha.device)  # Starting from pure noise (x_T)
        
        # Step 2: Reverse diffusion process (from step T to 0)
        for t in reversed(range(self.steps)):
            # Get current time as a tensor
            time_tensor = torch.tensor([t] * img_shape[0], device=self.alpha.device)
            
            # Sample the previous step (p(x_{t-1} | x_t))
            xt, _ = self.sample_p(xt, time_tensor)
        
        # Step 3: Return the denoised image (x0 approximation)
        return xt
        
        