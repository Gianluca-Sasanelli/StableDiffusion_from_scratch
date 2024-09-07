#Sample
from utils import CelebA
import torch, os, pickle, torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from model import *
import imageio  # For creating a video
import os
##Train parameters
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
num_grids = 1
out_dir = "output\output_CelebA"
device = "cuda"
batch_size = 6
seed = 45
image_size = 128
torch.manual_seed(seed)
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
ctx = torch.cuda.amp.autocast() if torch.cuda.is_available() else torch.no_grad()

checkpoint = torch.load(ckpt_path, map_location="cuda")
checkpoint_model_args = checkpoint["model_args"]
model = DiffusionModel(**checkpoint_model_args)
state_dict = checkpoint['model']
model.load_state_dict(state_dict)
model.eval()
model.to(device)
data_dir = r"C:\Users\Gianl\Desktop\celeba\img_align_celeba"
transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
data = CelebA(data_dir, transform = transform, size = 1.0) # with size you can specify if you want the whole dataset or part of it
    
train_split = int(len(data) * 0.9)
val_split = len(data) - train_split
train_set, val_set = torch.utils.data.random_split(data, [train_split, val_split])
train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True, pin_memory = True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle = False, pin_memory = True)
frames_dir = './frames'
os.makedirs(frames_dir, exist_ok=True)
def get_batch(split):
    if split == "train":
        batch = next(iter(train_loader))
    if split == "val":
        batch = next(iter(val_loader))
    return batch.to(device)
num_timesteps = 50 -1 
with torch.no_grad():
    batch = get_batch("val")  # 6 CelebA images
    for t_step in range(num_timesteps):
        # Sample images for the current timestep (assume you have these methods in your model)
        t = torch.tensor([t_step] * batch_size).to(device)
        
        print(t.shape)
        
        result, latent = model.visualization(batch, t2= t)
        
        # Create a grid of images (6 images in 1 row)
        grid = torchvision.utils.make_grid(result, nrow=2, normalize=True)
        
        # Save the grid as a frame (optional, you can skip this if you don't need individual images)
        grid_img = grid.permute(1, 2, 0).cpu().float().numpy()
        plt.imshow(grid_img)
        plt.axis('off')
        plt.savefig(f"{frames_dir}/frame_{t_step:03d}.png", bbox_inches='tight')
        plt.close()
    