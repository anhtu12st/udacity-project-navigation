import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    A simple fully-connected neural network for Q-value estimation in reinforcement learning.

    Parameters:
    - state_size (int): The dimension of the state space.
    - action_size (int): The dimension of the action space.
    - seed (int): Seed for reproducibility.

    Methods:
    - forward(state): Forward pass through the neural network.

    Attributes:
    - seed (int): Seed for reproducibility.
    - fc1 (Linear): The first fully connected layer with input size 'state_size' and output size 256.
    - fc2 (Linear): The second fully connected layer with input size 256 and output size 128.
    - fc3 (Linear): The third fully connected layer with input size 128 and output size 64.
    - fc4 (Linear): The fourth fully connected layer with input size 64 and output size 'action_size'.

    Example:
    ```python
    q_network = DQNetwork(state_size=8, action_size=4, seed=42)
    state = torch.randn(1, 8)  # Example input state
    q_values = q_network(state)
    print(q_values)
    ```
    """
    def __init__(self, state_size, action_size, seed=0):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)
        
    def forward(self, state):
        """Forward pass through the neural network."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
