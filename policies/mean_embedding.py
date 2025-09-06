import torch
from torch import nn
from torch.utils.data import DataLoader

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define model
class MeanEmbedding(nn.Module):
    def __init__(self, hidden_sizes, obs_size):
        super(MeanEmbedding, self).__init__()

        self.nr_layers = len(hidden_sizes)
        self.obs_size = obs_size
        self.network = nn.ModuleList()

        self.last_out = obs_size

        #Building the Mean-Embedding Network
        for i in range(self.nr_layers):
            self.network.append(nn.Linear(self.last_out, hidden_sizes[i]))
            self.network.append(nn.ReLU())
            self.last_out = hidden_sizes[i]

    def forward(self, input):
        #Assumption: The input data is the observation of one single agent
        logits = self.network(input)
        return logits 

model = MeanEmbedding(5, [10, 20, 5], 15, 5).to(device)
print(model)