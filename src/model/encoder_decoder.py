import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderDecoderModel(nn.Module):
    def __init__(self, input_channels=3, min_size=32):
        super(EncoderDecoderModel, self).__init__()
        self.min_size = min_size
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((None, None))
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Upsample if the input is smaller than the minimum size
        if x.shape[-2] < self.min_size or x.shape[-1] < self.min_size:
            scale_factor = self.min_size / min(x.shape[-2], x.shape[-1])
            x = F.interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=True)

        z = self.encoder(x)
        return self.decoder(z)