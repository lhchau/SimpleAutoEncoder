import torch
import torch.nn.functional as F
import torch.nn as nn 

class Decoder(nn.Module):
    def __init__(self, input_shape=[3, 227, 227]) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=2048),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=2048, out_features=3*227*227),
            nn.Sigmoid())
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = torch.reshape(out, [-1] + self.input_shape)
        return out
        