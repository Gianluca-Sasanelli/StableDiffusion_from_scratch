import torch, os
from matplotlib.animation import FuncAnimation
import torch.nn as nn
import torch.nn.functional as F
from model import *
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from utils import CelebA
from PIL import Image
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = "cuda"
data_dir = r"C:\Users\Gianl\Desktop\img_align_celeba" #directory of images
transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.CenterCrop(32),
            transforms.ToTensor()        ])

out_dir = "output/output_CelebAUnet32"
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location="cuda")
checkpoint_model_args = checkpoint["model_args"]
model = DiffusionModel(**checkpoint_model_args)
state_dict = checkpoint['model']
model.load_state_dict(state_dict)
model.eval()
model.to(device)
data = CelebA(data_dir, transform = transform, size = 1.0) # with size you can specify if you want the whole dataset or part of it
batch_size = 1
train_split = int(len(data) * 0.9)
val_split = len(data) - train_split
train_set, val_set = torch.utils.data.random_split(data, [train_split, val_split])
train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True, pin_memory = True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle = False, pin_memory = True)
print("Length train loader:", len(train_loader))
def get_batch(split):
    if split == "train":
        batch = next(iter(train_loader))
    if split == "val":
        batch = next(iter(val_loader))
    return batch.to(device)
class ImageGenerator:
    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def generate_image(self, img_shape, interval, steps=999, starting_noise=None):
        if starting_noise is None:
            xt = torch.randn(img_shape, device=device)
        else:
            xt = starting_noise
        time_steps = []
        images = []
        for t in reversed(range(steps)):
            time_tensor = torch.tensor([t] * img_shape[0], device=device)
            if t % interval == 0 or t==0:
                print(t)
                time_steps.append(t)
                images.append(self.process_image(xt[0]))
            xt, _ = self.model.sample_p(xt, time_tensor)
        
        return images, time_steps

    def process_image(self, img):
        img = img.clip(0, 1)
        img = img.cpu().numpy()
        return img.transpose(1, 2, 0)

    def create_animation(self, images,  time_steps):
        
        fig, ax = plt.subplots()
        plt.close()  # Prevents the empty figure from being displayed
        
        def animate(i):
            ax.clear()
            ax.imshow(images[i])
            ax.set_title(f"Step: {time_steps[i]}")
            ax.axis('off')
        
        anim = FuncAnimation(fig, animate, frames=len(images), interval=200, repeat_delay=5000)
        return anim

    def create_image_grid(self,  images,  time_steps, grid_size=(5, 5)):        
        rows, cols = grid_size
        fig, axs = plt.subplots(rows, cols, figsize=(15, 15))
        
        for i, ax in enumerate(axs.flat):
            if i < len(images):
                ax.imshow(images[i])
                ax.set_title(f"Step: {time_steps[i]}")
            ax.axis('off')
        
        plt.tight_layout()
        return fig

# Usage example:
generator = ImageGenerator(model)
interval = 50
# Get a batch and create starting noise
x = get_batch("val")
steps = 750
t = torch.tensor([steps] * x.shape[0], device=device)
starting_noise,_ = model.sample_q(x, t)

# Create an animation
images, time_steps = generator.generate_image((1, 3, 32, 32), interval = interval, steps = steps, starting_noise=starting_noise) 
anim = generator.create_animation(images,  time_steps)

# Save the animation
anim.save('celeba_generation_animation.gif', writer='pillow', fps=10)
print("Animation has been saved as 'celeba_generation_animation.gif'")

# Display the animation (this will work in a Jupyter notebook)

# Create an image grid
fig = generator.create_image_grid(images = images, time_steps = time_steps)
fig.savefig('celeba_generation_grid.png')
print("Image grid has been saved as 'celeba_generation_grid.png'")

# Display the image grid
plt.show()

# Close all plt figures to avoid backend issues
plt.close('all')