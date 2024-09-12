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
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_steps: int = None, dropout: float = 0.0,  sec_out_divisor: int = 1):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels , stride= 1, padding = 1,kernel_size = 3) 
        self.act1 = nn.SiLU()
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels // sec_out_divisor , stride= 1, padding = 1,kernel_size = 3)    
      
        if time_steps is not None:
            self.time_emb = nn.Linear(time_steps, out_channels)                      
        if in_channels == out_channels and sec_out_divisor == 1:
            self.conv_shortcut = nn.Identity()
            
        else:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels // sec_out_divisor, kernel_size=3, stride=1, padding = 1)
            
        self.dropout = nn.Dropout(dropout)        
        
    def forward(self, input:torch.Tensor, t:torch.Tensor):
        x = self.conv1(self.act1(self.norm1(input)))
        t = self.time_emb(t)[:,:,None,None]
        #adding the time embeddings
        x += t
        x= self.conv2(self.act1(self.norm2(self.dropout(x))))
        x += self.conv_shortcut(input)
        return x
    
#Attention layer in the latent space
class Attention_Block(nn.Module):
    def __init__(self, n_channels: int, n_heads:int = 1, dropout: float = 0.1):
        super().__init__()
        self.n_channels = n_channels
        #projection in queries, keys and values
        self.projections = nn.Linear(n_channels, n_channels * 3* n_heads)
        self.output_projection = nn.Linear(n_channels * n_heads, n_channels)
        flash = hasattr(nn.functional, "scaled_dot_product_attention")
        if not flash:
            print("Pytorch flash is not working... You should implement an manual attention mechanism")
        self.out_dropout = nn.Dropout(dropout)
        self.dropout = dropout
        
    def forward(self, x: torch.Tensor):
        b, c, height, width = x.shape
        res = x.view(b,c,-1).transpose(2,1)
        q,k,v = self.projections(res).split(c, dim = -1)
        res = F.scaled_dot_product_attention(q,k,v, 
                                           attn_mask=None, 
                                           dropout_p=self.dropout 
                                           if self.training else 0,
                                                                    is_causal=True)
        res = self.out_dropout(self.output_projection(res))
        res = res.transpose(1,2).contiguous().view(b,c,height,width)
        
        return x + res
        
#Horizontal block in the UNET
class HorBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_steps: int, is_attn: bool= False, dropout= None, sec_out_divisor : int = 1):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_steps, dropout = dropout, sec_out_divisor= sec_out_divisor)
        if is_attn:
            self.attn = Attention_Block(out_channels // sec_out_divisor, dropout = dropout )
        else:
            self.attn = nn.Identity()
    def forward(self, x:torch.Tensor, t: torch.Tensor):
        x = self.res(x,t)
        x = self.attn(x)
        return x
    
#Downsapling with conv2 with stride 2
class DownSample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels , stride= 2, padding = 1,kernel_size = 3)
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.act1 = nn.SiLU()
    def forward(self, x: torch.Tensor, t:torch.Tensor):
        _= t
        x = self.conv1(self.act1(self.norm1(x)))
        return x
#Upsample with transposted conv
class UpSample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, in_channels , stride= 2, padding = 1,kernel_size = 3, output_padding= 1)
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.act1 = nn.SiLU()
    def forward(self, x: torch.Tensor):
        x = self.conv1(self.act1(self.norm1(x)))
        return x
    
#Middle block at the bottom of the UNET
class Middle_block(nn.Module):
    def __init__(self, in_channels: int, time_steps: int, dropout: float = 0.0):
        super().__init__()
        self.res1 = ResidualBlock(in_channels, in_channels, time_steps, dropout)
        self.attn = Attention_Block(in_channels, dropout = dropout)
        self.res2 = ResidualBlock(in_channels, in_channels, time_steps, dropout)
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x,t)
        x = self.attn(x)
        x = self.res2(x,t)
        return x
class UNET(nn.Module):
     def __init__(self, in_channels: int,  proj_channels: int = 64, is_attn: list = None, multipliers: list = None, dropout: float = 0.1, time_steps:int = 1000):
        super().__init__()  
        """ This is a UNNET module. The output is an image with the same shape of the input.
        The time steps are embedded into a 2048 vector and then applied at any level both in the descending and ascending block.
            :param in_channels: (int) Channels of the input. 3 if RGB
            :param proj_channels: (int) First projections of the input
            :param is_attn : (list) List of bools indicating whether an attention should be applied at that position 
            :param multipliers : (list) List multipliers of the channels used in deeper layers of the model 
            :param dropout : (float) dropout to be applied in the residual layers 
            :param time_steps : (int) time_steps
        """
        self.in_channels = in_channels
        self.time_emb = nn.Embedding(time_steps, 2048) 
        self.time_steps = 2048
        self.img_proj = nn.Conv2d(in_channels, proj_channels, kernel_size=3, padding=1, stride = 1)
        if multipliers is None:
            self.multipliers = [1,2,2,4]
        else:
            self.multipliers = multipliers
        if is_attn is None:
            is_attn = [False, False, False, False]
        self.hidden_dims = [i*proj_channels for i in self.multipliers]
        self.hidden_dims.insert(0, proj_channels)
        #
        downpass = []
        for i in range(len(self.hidden_dims)-1):
            in_channels = self.hidden_dims[i]
            out_channels = self.hidden_dims[i + 1]
            downpass.append(HorBlock(in_channels=in_channels, out_channels=out_channels, dropout = dropout, time_steps= 2048, is_attn=is_attn[i]))
            downpass.append(DownSample(out_channels))
        self.downpass = nn.ModuleList(downpass)
        self.middle = Middle_block(self.hidden_dims[-1], time_steps= self.time_steps, dropout = dropout)
        up = []
        for i in reversed(range(1, (len(self.hidden_dims)))):
            in_channels = self.hidden_dims[i]
            up.append(UpSample(in_channels = in_channels))
            in_channels = in_channels * 2
            out_channels = self.hidden_dims[i-1] * 2
            up.append(HorBlock(in_channels = in_channels,out_channels=out_channels, time_steps= self.time_steps, dropout=dropout, sec_out_divisor = 2, is_attn= is_attn[i -1]))
        self.uppass = nn.ModuleList(up)
        
        #Last layer
        self.out_norm = nn.BatchNorm2d(self.hidden_dims[0])
        self.out_act = nn.SiLU()
        self.out_projections = nn.Conv2d(self.hidden_dims[0]*2 , self.in_channels, kernel_size=3, stride = 1, padding = 1)
        self.apply(self._init_weights)
        
        
     def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)   
            elif isinstance(module, nn.Conv2d):
                #I find it better than xavier initialization
                torch.nn.init.normal_(module.weight, mean=0.0, std = 0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.ConvTranspose2d):
                torch.nn.init.normal_(module.weight, mean=0.0, std = 0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
                    
                    
                    
     def encoder(self, x: torch.Tensor, t: torch.Tensor = None):
          residuals = [x]
          i = 0
          while i < len(self.downpass): 
               res = self.downpass[i](x, t)
               x = self.downpass[i+1](res, t)
               residuals.append(res)
               i += 2
          return x, residuals
     
     def decoder(self, x: torch.Tensor, residuals: list, t: torch.Tensor):
          i = 0
          residuals.reverse()
          while i < len(self.uppass):
               x = self.uppass[i](x)
               x = torch.cat((x, residuals[i // 2]), dim = 1) #concat on feature dimension   
               x = self.uppass[i+1](x, t)
               i +=2
          x = torch.cat((x, residuals[-1]), dim =1)
          x = self.out_projections(x)
          return x
     
     def forward(self, x:torch.Tensor, t: torch.Tensor):
          time_embeddings = self.time_emb(t)
          x = self.img_proj(x)
          x, residuals = self.encoder(x, time_embeddings)
          x = self.middle(x, time_embeddings)
          x = self.decoder(x= x, residuals = residuals, t= time_embeddings)
          return x

      
     def configure_optimizers(self, learning_rate):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        return optimizer    
                           


"""" Let's try to reproduce the DDPM paper: https://arxiv.org/pdf/2006.11239.
     Basically, we add noise to the data distribution q(xt|xt-1) and then we learn the distribution p(xt-1|xt).
     We cannot just reverse q(xt-1|xt) because it is intractable. 
     However, we can reverse q(xt-1| xt,x0), the reverse conditioned on the previous step and x0.
     Basically, this reverse can be written as function of the reconstructed noise.
     Now, defining the ELBO loss to minimize the difference between q reverse and p, which in practise is an MSE loss between the added noise
     at time step t and the reconstructed noise at the same time step.
     
     The model that gives us the eps_theta should work on images such as a Unet."""
class DiffusionModel(nn.Module):
    def __init__(self, in_channels:int = 3,
                 multipliers = None, proj_channels = 64, beta_min: float= 0.0001,
                 beta_max: float = 0.02, steps:int = 1000, 
                 is_attn: list = None, dropout: float = 0.1, device = "cuda" ):
        super().__init__()
        #The model that returns the noise which is with the same shape of the image
        self.reconstructed_noise = UNET(in_channels = in_channels, proj_channels=proj_channels, 
                                         time_steps= steps, multipliers=multipliers, 
                                         is_attn = is_attn, dropout= dropout)
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
    
    
    """Computing the loss is between the added noise and the reconstructed noise """
        
    def loss(self, x0:torch.Tensor, noise = None):
        batch_size = x0.shape[0]
        t = torch.randint(low = 0, high = self.steps, size= (batch_size,), device = x0.device, dtype = torch.long )
        if noise is None:
            noise = torch.randn_like(x0)
        xt, _ = self.sample_q(x0, t, eps = noise)
        eps_theta = self.reconstructed_noise(xt, t)
        return F.mse_loss(noise, eps_theta)
    
    def num_parameters(self):
        return sum(p.numel() for p in self.reconstructed_noise.parameters() if p.requires_grad)
    
    def config_optimizer(self, lr):
        optimizer = self.reconstructed_noise.configure_optimizers( learning_rate = lr)
        return optimizer
    
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
        
        