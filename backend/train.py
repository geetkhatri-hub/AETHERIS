import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.generator import AttentionUNetGenerator
from models.discriminator import PatchGANDiscriminator
from dataset import FLIRDataset

def train(args):
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Ensure checkpoints directory exists
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Initialize Dataset and DataLoader
    dataset = FLIRDataset(root_dir=args.dataset_path, split="train")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    # Initialize Networks
    generator = AttentionUNetGenerator(in_channels=1, out_channels=3).to(device)
    discriminator = PatchGANDiscriminator(in_channels=4).to(device)
    
    # Loss functions
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()
    
    # Optimizers (Standard Pix2Pix setup: lr=0.0002, beta1=0.5, beta2=0.999)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    # Hyperparameters
    lambda_L1 = args.lambda_L1 # Usually 100
    
    print("Starting Training Loop...")
    for epoch in range(1, args.epochs + 1):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        
        for i, (real_ir, real_rgb) in enumerate(dataloader):
            real_ir = real_ir.to(device)
            real_rgb = real_rgb.to(device)
            
            batch_size = real_ir.size(0)
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            # Real loss
            pred_real = discriminator(real_ir, real_rgb)
            loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real, device=device))
            
            # Fake loss
            fake_rgb = generator(real_ir)
            pred_fake = discriminator(real_ir, fake_rgb.detach())
            loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake, device=device))
            
            # Total Discriminator Loss
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()
            
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # Adversarial Loss (Generator wants to trick Discriminator)
            pred_fake = discriminator(real_ir, fake_rgb)
            loss_G_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake, device=device))
            
            # Pixel-wise L1 Loss
            loss_G_L1 = criterion_L1(fake_rgb, real_rgb)
            
            # Combined Loss
            loss_G = loss_G_GAN + (lambda_L1 * loss_G_L1)
            loss_G.backward()
            optimizer_G.step()
            
            # Accumulate loss
            epoch_g_loss += loss_G.item()
            epoch_d_loss += loss_D.item()
            
            if i % args.log_interval == 0:
                print(f"[Epoch {epoch}/{args.epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[D Loss: {loss_D.item():.4f}] [G Loss: {loss_G.item():.4f}] [G L1: {loss_G_L1.item():.4f}]")
        
        avg_g_loss = epoch_g_loss / len(dataloader)
        avg_d_loss = epoch_d_loss / len(dataloader)
        print(f"--> Epoch {epoch} Complete. Average D Loss: {avg_d_loss:.4f}, Average G Loss: {avg_g_loss:.4f}")
        
        # Save Checkpoint
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            gen_path = os.path.join(args.checkpoint_dir, f"generator_epoch_{epoch}.pth")
            disc_path = os.path.join(args.checkpoint_dir, f"discriminator_epoch_{epoch}.pth")
            torch.save(generator.state_dict(), gen_path)
            torch.save(discriminator.state_dict(), disc_path)
            print(f"Saved checkpoints to:\n  {gen_path}\n  {disc_path}")
            
    # Save Final weights as production standard names
    torch.save(generator.state_dict(), os.path.join(args.checkpoint_dir, "generator.pth"))
    torch.save(discriminator.state_dict(), os.path.join(args.checkpoint_dir, "discriminator.pth"))
    print("Training Complete. Final weights saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IR Satellite/Thermal Colorization Training (cGAN)")
    parser.add_argument("--dataset_path", type=str, default="dataset", help="Path to FLIR dataset root directory")
    parser.add_argument("--checkpoint_dir", type=str, default="models", help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=type(1), default=50, help="Number of training epochs (50-100 recommended)")
    parser.add_argument("--batch_size", type=type(1), default=4, help="Batch size for training")
    parser.add_argument("--lr", type=type(1.0), default=0.0002, help="Learning rate for Adam optimizer")
    parser.add_argument("--lambda_L1", type=type(1.0), default=100.0, help="L1 loss scaling factor")
    parser.add_argument("--num_workers", type=type(1), default=0, help="Number of data loading workers")
    parser.add_argument("--log_interval", type=type(1), default=10, help="Interval to log batch progress")
    parser.add_argument("--save_interval", type=type(1), default=10, help="Interval to save model weights")
    
    args = parser.parse_args()
    train(args)
