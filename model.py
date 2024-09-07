import torch
import torch.nn as nn
import torch.nn.functional as F

""""Main ideas behind the paper/model https://arxiv.org/pdf/2112.10752:
1)Doing the diffusion process on a lower dimensional space (obtained 
with an autoencoder). It is computationally more efficient than doing it on pixel levels.
2) I don't understand it now
3)Obtaining "general-purpose" compression models whose latent space
can be used to train multiple generative models.

The autoencoder is trained with a perceptual loss and a patch-based adversarial objective.

What is a diffusion model https://arxiv.org/pdf/2006.11239? https://lilianweng.github.io/posts/2021-07-11-diffusion-models/

Probabilistic model that learn the data distribution p(x)
by denoising a normally distributed variable. The denoising consists in a process of fixed markov chain length T
The loss would be an MSE between the expected noise by the model and the real noise.
The neural backbone that predicts the noise is a UNET.

How do we conditionit on text for instance?
Diffusion model are capble of learning a conditional distribution of the type p(z|y).
The idea is doing it through an cross attention mechanism. The idea is to transform the (text)input y
into t_theta (y), through an autoencoder and then apply a cross attention somewhere in the UNET.

"""

#Let's start with a UNET
#The idea is to have (128)^2 pixel images 
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
        
    def forward(self, x: torch.Tensor,t: torch.Tensor = None):
        x = self.act1(self.norm1(self.conv1(x)))
        x += self.time_emb(t)[:,:,None, None]
        res = self.act2(self.norm2(self.conv2(x)))
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
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512, 1024]
        hidden_dims.insert(0, in_channels) 
        
        self.final_hidden_dim = hidden_dims[-1]
        self.downpass = nn.ModuleList(
            [DownSample(in_channels = hidden_dims[i], out_channels = hidden_dims[i+1], time_steps= time_steps) for i in range(len(hidden_dims) -1 )]) 
           
        self.enc_lin = nn.Linear(self.final_hidden_dim*4, latent_dim)
        self.dec_lin = nn.Linear(latent_dim, self.final_hidden_dim * 16)
        
        hidden_dims.reverse()
        self.uppass = nn.ModuleList(
            [UpSample(in_channels = hidden_dims[i]* 2) for i in range(len(hidden_dims) - 2 )])
        self.final_layer = nn.Sequential( 
                                         nn.Conv2d(in_channels= hidden_dims[-2] , out_channels=hidden_dims[-1], kernel_size=3,
                                                   stride=1, padding=1),
                                         nn.Tanh()
                                         )
            
        hidden_dims.reverse()
        
    def encoder(self, x: torch.Tensor, t:torch.Tensor = None):
        residuals = []
        for down in self.downpass:
            x, res = down(x, t)
            residuals.append(res)
        x = torch.flatten(x, start_dim= 1)
        x = self.enc_lin(x)
        return x, residuals
    
    def decoder(self, x: torch.Tensor, residuals: list):
        x = self.dec_lin(x)
        x = x.view(-1, self.final_hidden_dim, 4, 4)
        for i, up in enumerate(self.uppass):
            #concatenating the residuals on the feature dimension
            # print(f"At up pass step {i} residuals shape is {residuals[i].shape}")
            x = torch.cat((x, residuals[i]), dim = 1)
            # print(f"At up pass step {i} x shape is {x.shape}")
            x = up(x)
            # print(f"At the end of the up pass step {i} x shape is {x.shape}")
        return x
    
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x, residuals = self.encoder(x, t)
        residuals.reverse()
        x = self.decoder(x, residuals)
        x = self.final_layer(x)
        return x

"""" Let's try to reproduce the DDPM paper: https://arxiv.org/pdf/2006.11239.
     Basically, we add noise to the initial q(x0) and then we learn the distribution p to reverse the problem.
     We cannot just reverse q(xt-1|xt) because it is intractable. However, we can reverse q(xt-1| xt,x0), the reverse conditioned on the previous step and x0.
     Basically, this reverse is given by exp(-1/2 B_tilde(xt) * x^2_t-1 - mu_tile(xt,x0)* x_t-1).
     Given that x0 is known and depends on eps(we can sample at any t): mu(x_t, eps)_tilde formula is known.
     Now, defining the ELBO loss to minimize the difference between q reverse and p.
     So, to recap, we must learn mu_theta of p to predict mu_tilde(x_t, epsilon_t). Because,
     x_t is known we just need to predict epsilon. The final loss can be simpified to:
     L_t = E[MSE(eps_t - eps_theta(x_t,t))].
     The model that gives us the eps_theta should work on images such as a Unet."""
class DiffusionModel(nn.Module):
    def __init__(self, beta_min: float= 0.0001, beta_max: float = 0.02, steps:int = 50 ):
        super().__init__()
        #number of time steps
        self.eps_model = UNET(in_channels = 3, latent_dim= 256, time_steps= steps)
        self.steps = steps
        #variance scheduler
        self.beta = torch.linspace(beta_min, beta_max, steps)
        #necessary to compute the mean of q
        self.alpha = 1 - self.beta
        self.alpha_tot = torch.cumprod(self.alpha, dim = 0)
    """Adding noise to the distribution, sampling at any t with the reparametrization trick
    q(xt|x0) = N(xt; alpha_tot^0.5 * x0, (1-alpha_tot) * I)
    """
    def sample_q(self, x0: torch.Tensor, t: torch.Tensor, eps= None):
        mean = self.alpha_tot[t].view(-1,1,1,1) ** 0.5 * x0  
        var =  1 - self.alpha_tot[t].view(-1,1,1,1)
        if eps is None:
            eps = torch.randn_like(x0)
        sample = mean + (var ** 0.5) * eps        
        return sample
    """denoising the distribution: The goal is to learn the distribution p which in teory is the reverse of q
    p(xo:t) = p(xt) \prod p(xt-1|xt) --> p(xt-1|xt ) = N(xt-1; mu(xt,t), var(xt, t))
    p(xt) = N(xt;0,I).
    """
    def sample_p(self, xt: torch.Tensor, t: torch.Tensor):
        eps_theta = self.eps_model(xt, t)
        alpha_bar = self.alpha_tot[t].view(-1,1,1,1)
        alpha = self.alpha[t].view(-1,1,1,1)
        coef = (1 -alpha) / (1- alpha_bar) ** 0.5
        mean = 1 / (alpha ** 0.5) * (xt - coef * eps_theta)
        var = self.beta[t].view(-1,1,1,1)
        eps = torch.randn(xt.shape, device = xt.device)
        return mean + (var ** .5) * eps
    """ The loss is between the noise and epsilon """
    def loss(self, x0:torch.Tensor, noise = None):
        batch_size = x0.shape[0]
        t = torch.randint(low = 0, high = self.steps, size= (batch_size,), device = x0.device, dtype = torch.long )
        if noise is None:
            noise = torch.randn_like(x0)
        xt = self.sample_q(x0, t, eps = noise)
        eps_theta = self.eps_model(xt, t)
        return F.mse_loss(noise, eps_theta)
        