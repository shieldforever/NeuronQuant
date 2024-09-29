from unicodedata import numeric
import torch.nn as nn

class MLP_MNIST(nn.Module):
    def __init__(self):
        super(MLP_MNIST, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=28*28, out_features=500)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=500, out_features=10)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        out = self.flatten(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def mlp_mnist():
    return MLP_MNIST()
