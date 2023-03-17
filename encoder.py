import torch
import torch.nn.functional as F
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, alex, latent_size=1024) -> None:
        super().__init__()
        self.alex = alex
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=latent_size),
            nn.ReLU())
        
    def forward(self, x):
        """
        After AlexNet we have feature map with (4096 * 4096)
        """
        out = self.alex(x)
        out = self.fc2(out)
        return out        