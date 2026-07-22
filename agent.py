import torch
import torch.nn as nn
from rocket_env import RocketEnv

env = RocketEnv()

class RocketNetwork(nn.Module):

    def __init__(self, state_size, action_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, action_size)
        )


    def forward(self, x):
        return self.network(x)
    


model = RocketNetwork(6,6)


obs, info = env.reset()


state = torch.tensor(obs, dtype=torch.float32)


with torch.no_grad():
    q_values = model(state)


action = torch.argmax(q_values).item()