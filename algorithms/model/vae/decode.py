from torch import nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=256, output_dim=512):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        recon = self.fc2(h)
        return recon