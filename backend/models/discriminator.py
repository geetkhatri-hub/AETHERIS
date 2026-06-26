import torch
import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator.
    Determines if a 70x70 patch in the image is real or fake.
    Takes the concatenation of the input (IR, 1 channel) and target/generated (RGB, 3 channels).
    So total input channels = 1 + 3 = 4.
    """
    def __init__(self, in_channels=4, ndf=64):
        super().__init__()
        
        # We define a standard Pix2Pix PatchGAN architecture
        self.model = nn.Sequential(
            # Layer 1: Conv -> LeakyReLU (no BatchNorm)
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2: Conv -> BatchNorm -> LeakyReLU
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3: Conv -> BatchNorm -> LeakyReLU
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4: Conv -> BatchNorm -> LeakyReLU (stride=1 to keep dimension larger)
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 5: Output Conv -> 1 channel prediction map
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)
            # Output represents logits (no Sigmoid here, we use BCEWithLogitsLoss for stability)
        )

    def forward(self, ir_img, rgb_img):
        # Concatenate along the channel dimension (dim=1)
        x = torch.cat([ir_img, rgb_img], dim=1)
        return self.model(x)

if __name__ == "__main__":
    # Test execution
    disc = PatchGANDiscriminator(in_channels=4, ndf=64)
    dummy_ir = torch.randn(1, 1, 256, 256)
    dummy_rgb = torch.randn(1, 3, 256, 256)
    
    output = disc(dummy_ir, dummy_rgb)
    print("Discriminator IR input shape:", dummy_ir.shape)
    print("Discriminator RGB input shape:", dummy_rgb.shape)
    print("Discriminator Output shape (patch logits):", output.shape)
    # Patch size for 256x256 input is usually 30x30 with this configuration
    assert len(output.shape) == 4 and output.shape[1] == 1, "Output should have shape (N, 1, H_patch, W_patch)"
    print("Discriminator verification passed successfully!")
