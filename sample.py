import torch, torchvision,os
import torch.nn as nn
import torch.nn.functional as F
from model import *
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from utils import CelebA
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = "cuda"
data_dir = r"C:\Users\Gianl\Desktop\img_align_celeba" #directory of images
transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406] ,std= [0.229, 0.224, 0.225]) 
        ])

out_dir = "output/output_CelebAUnet2"
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location="cuda")
checkpoint_model_args = checkpoint["model_args"]
model = DiffusionModel(**checkpoint_model_args)
state_dict = checkpoint['model']
# model.load_state_dict(state_dict)
# model.eval()
# model.to(device)
print(checkpoint_model_args["hidden_dims"]) 

