#train
import os, datetime, torch, pickle,  argparse, yaml
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from model import *
from utils import CelebA, galaxy_Dataset


# #Parser. Use --config $ConfigPath. Reading from yaml config file
# parser = argparse.ArgumentParser()
# parser.add_argument("--config", type = str, required = True, help = "config path")
# args = parser.parse_args()
# config_path = args.config
# with open(config_path) as file:
#     config = yaml.safe_load(file)
    
# #data
device = "cuda"
data_dir = r"C:\Users\Gianl\Desktop\celeba\img_align_celeba" #directory of images


# Parameters
## Settings
image_size = 128
size_dataset = 1.0
precision = torch.bfloat16
##logging parameters
out_dir = "output/output_CelebA"
os.makedirs(out_dir, exist_ok=True)
log_interval = 5
eval_iters = 10
init_from = "scratch"
num_epochs = 10
best_val_loss = 1e9


##model parameters
in_channels = 3
latent_dimension = 512
##Training parameters
batch_size = 128
max_lr = 0.005
gamma = 0.95
# gradient_accomulation_iter = 1
decay_lr = True
grad_clip = 1.0

#Defining the module
model_args = dict(in_channels = in_channels, latent_dim = latent_dimension)
#optimizer and scheduler
if init_from == "scratch":
    print("Initializing a new model from scratch")
    model = DiffusionModel(**model_args)
    resume_epoch = 0
elif init_from == "resume":
    print(f"Resuming from {out_dir}")
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location = device)
    checkpoint_model_args = checkpoint["model_args"]
    for k in ['in_channels', 'latent_dim', 'kld_weight', 'hidden_dims']:
        model_args[k] = checkpoint_model_args[k]
    model = VanillaVAE(**model_args)
    state_dict = checkpoint["model"]
    model.load_state_dict(state_dict)
    resume_epoch = checkpoint["epoch"]
    best_val_loss = checkpoint["best_val_loss"]
    
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = max_lr)
if init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None #flush the memory

#useful functions
transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

def get_batch(split):
    if split == "train":
        batch = next(iter(train_loader))
    if split == "val":
        batch = next(iter(val_loader))
    return batch.to(device)

def get_lr(iter):
        return max_lr * (gamma ** iter)
    
#Initialization of train and val set
# ------------------------------------
data = CelebA(data_dir, transform = transform, size = size_dataset) # with size you can specify if you want the whole dataset or part of it
    
train_split = int(len(data) * 0.9)
val_split = len(data) - train_split
train_set, val_set = torch.utils.data.random_split(data, [train_split, val_split])
train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True, pin_memory = True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size = batch_size, shuffle = False, pin_memory = True)
print("Length train loader:", len(train_loader))
# -------------------------------------
#scaler and ctx using bfloat16 data 
ctx = torch.amp.autocast(device_type = device, dtype = torch.bfloat16)
scaler = torch.cuda.amp.GradScaler(enabled = True)


@torch.no_grad()
#Valdiation pass
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        running_losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            data = get_batch(split)
            with ctx:
                loss = model.loss(data)
            running_losses[k] = loss.item()
        out[split] = running_losses.mean(dim = 0)
    model.train()
    return out
#Training pss
def one_epoch_pass():
    model.train()
    last_loss = torch.zeros((1,))
    t0 = datetime.datetime.now()
    for i, batch in enumerate(train_loader):
        data = batch.to(device)
        with ctx:
           loss= model.loss(data)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if i % log_interval == 0 and i != 0:
            last_loss= loss.item()
            t1 = datetime.datetime.now()
            dt = t1 -t0 
            print(f"step {i // log_interval}| loss {last_loss:.5f}| norm: {norm:.2f}| time {dt.seconds:.2f} s")
            t0 = t1
    return [last_loss]

#TRAINING LOOP
train_losses = []
val_losses = []
print("Starting of the training loop")
for epoch in range(resume_epoch, num_epochs):
    lr = get_lr(epoch) if decay_lr else max_lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print("-----------------------------------------")
    print("lr:", lr)
    t0 = datetime.datetime.now()
    losses = estimate_loss()
    val_losses.append(losses["val"])
    print(f"val | epoch {epoch}| train loss {losses['train']:.4f}| val  loss {losses['val']:.2e}")
    if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            if epoch > 0:
                checkpoint = {
                    "model" : model.state_dict(),
                    "optimizer" : optimizer.state_dict(),
                    "model_args": model_args,
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                }    
                print(f"save checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
            with open(out_dir + os.sep + "train_losses.pkl", "wb") as file:
                    pickle.dump(train_losses, file)
            with open(out_dir + os.sep +  "val_losses.pkl", "wb") as file:
                    pickle.dump(val_losses, file)       

    train_losses.extend(one_epoch_pass())
    t1 = datetime.datetime.now()
    dt = (t1 -t0)
    dt = dt.seconds / 60
    print(f"Epoch {epoch} ends, time of validation and training of one epochs: {dt:.1f} minutes")
    print("--------------------------------")
    if epoch > num_epochs - 1:
        with open(out_dir + os.sep + "train_losses.pkl", "wb") as file:
            pickle.dump(train_losses, file)
        with open(out_dir + os.sep +  "val_losses.pkl", "wb") as file:
            pickle.dump(val_losses, file)
        break
    
    