import yaml
import wandb
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import os
import torchvision

from src.dataset import CatDataset
from src.model_vae import MLP_VAE
from src.utils import pretty_now
from src.utils import compute_log_likelihood_iwae, reconstruction_log_likelihood
from src.utils import save_grid
from src.utils import print_stats
import numpy as np
from torchvision.transforms import v2

from dotenv import load_dotenv
load_dotenv()



def save_sample_images(model, save_path, cfg):
    device = next(model.parameters()).device
    z_samples = torch.zeros(25, 1, cfg['model']['d']).normal_(0.0, 1.0, generator=torch.manual_seed(0))
    z_samples = z_samples.to(device)
    images = model.generative_distr(z_samples).detach().squeeze(1)
    inverse_transform = v2.Compose(
            [ 
                v2.Normalize(mean=[ 0., 0., 0. ], std=list(1 / np.array([0.229, 0.224, 0.225]))),
                v2.Normalize(mean=list(-1 * np.array([0.485, 0.456, 0.406])), std=[ 1., 1., 1. ]),
            ]
        )
    images = inverse_transform(images)
    grid = torchvision.utils.make_grid(images, 5)
    save_grid(grid, save_path)

def training_epoch(model, optimizer, loader, epoch_size, tqdm_desc):
    device = next(model.parameters()).device
    mean_vlb = 0.0
    seen_objects = 0
    model.train()
    
    for step_num, images in tqdm(enumerate(loader), total=min(epoch_size, len(loader)), desc=tqdm_desc):
        if step_num == epoch_size:
            break
        optimizer.zero_grad()
        images = images.to(device)
        
        minus_vlb = -1 * model.batch_vlb(images)
        if torch.isnan(minus_vlb).any():
            print("NAN!!")
            continue
        minus_vlb.backward()
        optimizer.step()
        
        mean_vlb += -1 * minus_vlb.item() * images.shape[0]
        wandb.log({'batch_vlb': -1 * minus_vlb.item()})
        seen_objects += images.shape[0]
    mean_vlb /= seen_objects
    return mean_vlb

def validation_epoch(model, loader, tqdm_desc):
    device = next(model.parameters()).device
    mean_ll = 0.0
    model.eval()
    for images in tqdm(loader, desc=tqdm_desc):
        images = images.to(device)
        ll = compute_log_likelihood_iwae(images, model, 10)
        mean_ll += ll
    mean_ll /= len(loader.dataset)
    return mean_ll

def train(model, optimizer: torch.optim.Optimizer, train_loader: DataLoader, val_loader: DataLoader, cfg):   
    num_epochs = cfg['training']['num_epochs']
    
    for epoch in range(1, num_epochs + 1):
        vlb = training_epoch(
            model, optimizer, train_loader, cfg['training']['epoch_size'],
            tqdm_desc=f'Training {epoch}/{num_epochs}'
        )
        print(f"Epoch {epoch}/{num_epochs}. train_vlb: {vlb}")
        wandb.log({
            "epoch": epoch, 
            "epoch_vlb": vlb
        })

        val_ll = validation_epoch(model, val_loader, tqdm_desc=f'Validation {epoch}/{num_epochs}')
        wandb.log({'val_ll': val_ll})
        print(f"Epoch {epoch}/{num_epochs}. val_ll: {val_ll}")
        
        img_save_path = os.path.join(cfg['training']['save_path'], f'epoch_{epoch}.png')
        save_sample_images(model, img_save_path, cfg)
        wandb.log({"example": wandb.Image(img_save_path)})
        if cfg['training']['save_path'] is not None:
            torch.save(model.state_dict(), os.path.join(cfg['training']['save_path'], 'model_weights.pth'))



if __name__ == "__main__":
    config_path = '/home/ubuntu/image_generation/configs/config.yaml'
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg['training']['save_path'] = cfg['training']['save_path'].replace('{pretty_time}', pretty_now())
    cfg['training']['epoch_size'] = cfg['training']['epoch_size'] or int(1e8)
    print(cfg)
    
    os.mkdir(cfg['training']['save_path'])
        
    wandb.init(
        project="image_generation",
        config=cfg,
        # mode='disabled'
    )
        
    train_set = CatDataset(**cfg['dataset']['general'], **cfg['dataset']['train'])
    val_set = CatDataset(**cfg['dataset']['general'], **cfg['dataset']['val'])
    train_loader = DataLoader(train_set, batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=cfg['training']['batch_size'], shuffle=False, num_workers=4)

    
    
    device = torch.device('cuda')
    model = MLP_VAE(D=(3, 64, 64), **cfg['model']).to(device)
    print(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['lr'], weight_decay=cfg['training']['weight_decay'])
    
    train(model, optimizer, train_loader, val_loader, cfg)
    