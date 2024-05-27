import argparse
import os, sys
import torch
import numpy as np
from tqdm import tqdm
import datetime, random
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import Ambisonizer
from model.seanet import SEANet
from utils import seed_worker, set_seed
import auraloss

def kl_divergence(mu, log_var):
    """
    Computes the KL divergence between the learned latent distribution and a standard Gaussian distribution.
    Args:
        mu (torch.Tensor): Mean of the learned latent distribution.
        log_var (torch.Tensor): Log variance of the learned latent distribution.
    Returns:
        kl_div (torch.Tensor): KL divergence term.
    """
    log_var = torch.clamp(log_var, min=-10, max=10)
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - torch.exp(log_var), dim=1)
    return kl_div.mean()

def main(args):
    # Set the seed for reproducibility
    set_seed(42)
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    mrstft = auraloss.freq.MultiResolutionSTFTLoss().to(device)
    mse = nn.MSELoss().to(device)

    # Create the dataset
    path = args.base_dir
    train_dataset = Ambisonizer(path, partition="train")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=seed_worker)
    
    val_dataset = Ambisonizer(path, partition="val")
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, worker_init_fn=seed_worker)
    
    # Create the model
    model = SEANet(120000, args.embed_dim).to(device)

    # Create the optimizer
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.999), eps=1e-9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-7)

    # Create the directory for the logs
    log_dir = os.path.join(args.log_dir, str(args.embed_dim))
    os.makedirs(log_dir, exist_ok=True)
    
    # get current time
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(log_dir, current_time)
    os.makedirs(log_dir, exist_ok=True)

    # Create the summary writer
    writer = SummaryWriter(log_dir=log_dir)
    
    # Create the directory for the checkpoints
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save config for reproducibility
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        f.write(str(vars(args)))
    
    best_val_loss = float("inf")

    # Train the model
    for epoch in range(args.epochs):
        model.train()
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")):
            if args.debug and i > 20:
                break
            source, target = batch
            source = source.to(device)
            target = target.to(device)
            pred, mu, log_var = model(source)
            kl_loss = kl_divergence(mu, log_var)
            mrstft_loss = mrstft(pred, target)
            mse_loss = mse(pred, target)
            loss = mrstft_loss + mse_loss * 10 + kl_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            writer.add_scalar("Train/Loss", loss.item(), epoch * len(train_loader) + i)
            writer.add_scalar("Train/MRSTFT", mrstft_loss.item(), epoch * len(train_loader) + i)
            writer.add_scalar("Train/MSE", mse_loss.item(), epoch * len(train_loader) + i)
            writer.add_scalar("Train/KL", kl_loss.item(), epoch * len(train_loader) + i)
        scheduler.step()
        writer.add_scalar("LR/train", scheduler.get_last_lr()[0], epoch * len(train_loader) + i)
        
        model.eval()
        val_loss_dict = {
            'MRSTFT': 0,
            'MSE': 0,
            'KL': 0,
        }
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader, desc=f"Validation")):
                if args.debug and i > 20:
                    break
                source, target = batch
                source = source.to(device)
                target = target.to(device)
                pred, mu, log_var = model(source)
                kl_loss = kl_divergence(mu, log_var)
                mrstft_loss = mrstft(pred, target)
                mse_loss = mse(pred, target)
                val_loss_dict['MRSTFT'] += mrstft_loss.item()
                val_loss_dict['MSE'] += mse_loss.item()
                val_loss_dict['KL'] += kl_loss.item()
        val_loss_dict['MRSTFT'] /= len(val_loader)
        val_loss_dict['MSE'] /= len(val_loader)
        val_loss_dict['KL'] /= len(val_loader)
        val_loss = val_loss_dict['MRSTFT'] + val_loss_dict['MSE'] * 10 + val_loss_dict['KL']
        writer.add_scalar("Val/Loss", val_loss, epoch)
        writer.add_scalar("Val/MRSTFT", val_loss_dict['MRSTFT'], epoch)
        writer.add_scalar("Val/MSE", val_loss_dict['MSE'], epoch)
        writer.add_scalar("Val/KL", val_loss_dict['KL'], epoch)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
        # save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"epoch_{epoch + 1}.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True, help="The base directory of the dataset.")
    parser.add_argument("--epochs", type=int, default=5000, help="The number of epochs to train.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode; will only run for 20 steps for each epoch.")
    parser.add_argument("--gpu", type=int, default=0, help="The GPU to use.")
    parser.add_argument("--batch_size", type=int, default=32, help="The batch size for training.")
    parser.add_argument("--num_workers", type=int, default=8, help="The number of workers for the data loader.")
    parser.add_argument("--embed_dim", type=int, default=64, help="The number of workers for the data loader.")
    parser.add_argument("--log_dir", type=str, default="logs", help="The directory for the logs.")
    
    args = parser.parse_args()
    main(args)