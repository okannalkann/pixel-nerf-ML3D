import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from model.encoder_decoder import EncoderDecoderModel
from model.vgg_loss import VGGLoss 

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 16
learning_rate = 1e-3
num_epochs = 25

# Load your dataset
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(os.path.abspath('enco_deco_data/train'), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss, and optimizer
encoder_decoder = EncoderDecoderModel(input_channels=3).to(device)
vgg_loss_fn = VGGLoss().to(device)
l2_loss_fn = nn.MSELoss()
optimizer = optim.Adam(encoder_decoder.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    encoder_decoder.train()
    total_loss = 0
    for i, (inputs, _) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = inputs.clone()  # Autoencoder, so target is the same as input

        # Forward pass
        outputs = encoder_decoder(inputs)

        # Calculate losses
        l2_loss = l2_loss_fn(outputs, targets)
        vgg_loss = vgg_loss_fn(outputs, targets)
        loss = l2_loss + 0.01 * vgg_loss  # Adjust weight as needed

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

# Save the trained model
torch.save(encoder_decoder.state_dict(), 'encoder_decoder.pth')
print("Model saved as encoder_decoder.pth")