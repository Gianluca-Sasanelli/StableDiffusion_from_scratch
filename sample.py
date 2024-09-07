#Sample
from utils import CelebA
import torch, os, pickle, torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from model import *
##Train parameters
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
num_grids = 1
out_dir = "output\output_CelebA"
device = "cuda"
num_images = 6
seed = 45
image_size = 128
torch.manual_seed(seed)
ctx = torch.amp.autocast(device_type = "cuda", dtype = torch.bfloat16)
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
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
train_loader = torch.utils.data.DataLoader(train_set, batch_size = num_images, shuffle = True, pin_memory = True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size = num_images, shuffle = False, pin_memory = True)
print("Length train loader:", len(train_loader))
def get_batch(split):
    if split == "train":
        batch = next(iter(train_loader))
    if split == "val":
        batch = next(iter(val_loader))
    return batch.to(device)
with torch.no_grad():
    with ctx:
        fig, axes = plt.subplots(1, num_grids, figsize=(50 * num_grids, 40))
        for i in range(num_grids):
            _, xt, t = model.loss(get_batch("val"))
            print(t)
            samples = model.sample_p(xt = xt, t= t)


            grid = torchvision.utils.make_grid(samples, nrow=2, normalize=True)
            
            # Select the correct axis for the subplot
            ax = axes[i] if num_grids > 1 else axes
            
            ax.imshow(grid.permute(1, 2, 0).cpu().float().numpy())
            ax.axis('off')  # Remove axes for a cleaner look

        plt.show()