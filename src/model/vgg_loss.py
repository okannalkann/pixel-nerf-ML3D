import torch
import torch.nn as nn
import torchvision.models as models

class VGGLoss(nn.Module):
    def __init__(self, layer='relu3_3'):
        super(VGGLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.vgg_layers = nn.Sequential()
        for name, module in vgg._modules.items():
            self.vgg_layers.add_module(name, module)
            if name == layer:
                break
        
        # Freeze VGG parameters
        for param in self.vgg_layers.parameters():
            param.requires_grad = False
        
        self.criterion = nn.MSELoss()  # L2 Loss

    def forward(self, x, y):
        # Extract VGG features
        x_vgg = self.vgg_layers(x)
        y_vgg = self.vgg_layers(y)
        return self.criterion(x_vgg, y_vgg)