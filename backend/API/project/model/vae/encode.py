from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, latent_dim=128):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var