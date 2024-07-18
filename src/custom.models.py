import torch
import torch.nn as nn


class CustomEncoder(nn.Module):
    def __init__(self, input_height=300, input_width=400):
        super(CustomEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # Calculate the size after convolution layers
        conv_output_height = input_height // 16  # Assuming input_size is divisible by 16
        conv_output_width = input_width // 16  # Assuming input_size is divisible by 16
        self.fc = nn.Linear(512 * conv_output_height * conv_output_width, 1024)

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x


class CustomDecoder(nn.Module):
    def __init__(self, output_size=(300, 400)):
        super(CustomDecoder, self).__init__()
        self.fc = nn.Linear(1024, 512 * (output_size[0] // 16) * (output_size[1] // 16))
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (512, output_size[0] // 16, output_size[1] // 16)),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # to output values between 0 and 1
        )

    def forward(self, x):
        x = self.fc(x)
        x = self.decoder(x)
        return x
