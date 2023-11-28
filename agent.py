from collections import deque, namedtuple
import random
import torch
import numpy as np
from torch import optim
from model import QNetwork


class Agent:
    """
    An agent implementing a Deep Q-Network (DQN) for reinforcement learning.

    Parameters:
    - state_size (int): The dimension of the state space.
    - action_size (int): The dimension of the action space.
    - lr (float, optional): The learning rate for the optimizer. Default is 5e-4.
    - buffer_size (int, optional): The maximum capacity of the replay buffer. Default is 1e5.
    - batch_size (int, optional): The number of experiences to sample in each learning step. Default is 64.
    - update_every (int, optional): The frequency with which the target network is updated. Default is 4.
    - gamma (float, optional): The discount factor for future rewards. Default is 0.99.
    - tau (float, optional): The interpolation parameter for soft updates of target parameters. Default is 1e-3.
    - seed (int, optional): Seed for reproducibility. Default is 0.
    - device (str, optional): The device to which tensors should be moved ('cpu' or 'cuda'). Default is 'cpu'.

    Methods:
    - step(state, action, reward, next_state, done): Store the experience in the replay buffer and trigger the learning process.
    - act(state, eps=0.): Choose an action using epsilon-greedy policy.
    - learn(): Update the Q-network based on sampled experiences.
    - soft_update(): Perform a soft update of the target Q-network parameters.

    Attributes:
    - state_size (int): The dimension of the state space.
    - action_size (int): The dimension of the action space.
    - batch_size (int): The number of experiences to sample in each learning step.
    - update_every (int): The frequency with which the target network is updated.
    - gamma (float): The discount factor for future rewards.
    - tau (float): The interpolation parameter for soft updates of target parameters.
    - seed (int): Seed for reproducibility.
    - qnetwork_local (QNetwork): The local Q-network used for learning.
    - qnetwork_target (QNetwork): The target Q-network used for computing target values.
    - optimizer (Adam): The optimizer used for updating the Q-network.
    - memory (ReplayBuffer): The replay buffer for storing and sampling experiences.
    - t_step (int): Counter for tracking when to update the target network.

    Example:
    ```python
    agent = Agent(state_size=8, action_size=4, seed=42)
    state = np.random.randn(1, 8)  # Example input state
    action = agent.act(state)
    reward = 1.0
    next_state = np.random.randn(1, 8)  # Example next state
    done = False
    agent.step(state, action, reward, next_state, done)
    ```
    """
    def __init__(self,
                 state_size,
                 action_size,
                 lr=5e-4,
                 buffer_size=int(1e5),
                 batch_size=64,
                 update_every=4,
                 gamma=0.99,
                 tau=1e-3,
                 seed=0,
                 device='cpu',
                ):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.update_every = update_every
        self.gamma = gamma
        self.tau = tau
        self.seed = seed
        self.device = device
        
        self.qnetwork_local = QNetwork(self.state_size, self.action_size, seed).to(device)
        self.qnetwork_target = QNetwork(self.state_size, self.action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed, device=device)
        
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        """Store the experience in the replay buffer and trigger the learning process."""
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.update_every
        
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                self.learn()
    
    def act(self, state, eps=0.):
        """Choose an action using epsilon-greedy policy."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    
    def learn(self):
        """Update the Q-network based on sampled experiences."""
        states, actions, rewards, next_states, dones = self.memory.sample()
        
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return self.soft_update()
    
    def soft_update(self):
        """Perform a soft update of the target Q-network parameters."""
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1. - self.tau) * target_param.data)


class ReplayBuffer:
    """
    A simple replay buffer implementation for storing and sampling experiences in a reinforcement learning setting.

    Parameters:
    - action_size (int): The dimension of the action space.
    - buffer_size (int): The maximum capacity of the replay buffer.
    - batch_size (int): The number of experiences to sample in each batch.
    - seed (int): Seed for reproducibility.
    - device (str, optional): The device to which tensors should be moved ('cpu' or 'cuda'). Default is 'cpu'.

    Methods:
    - add(state, action, reward, next_state, done): Add a new experience to the replay buffer.
    - sample(): Sample a batch of experiences from the replay buffer.
    - __len__(): Return the current size of the replay buffer.

    Attributes:
    - action_size (int): The dimension of the action space.
    - memory (deque): A deque object representing the replay buffer.
    - batch_size (int): The number of experiences to sample in each batch.
    - experience (namedtuple): A named tuple representing an experience with fields: 'state', 'action', 'reward', 'next_state', and 'done'.
    - seed (int): Seed for reproducibility.
    - device (str): The device to which tensors should be moved ('cpu' or 'cuda').

    Example:
    ```python
    buffer = ReplayBuffer(action_size=4, buffer_size=10000, batch_size=32, seed=42, device='cuda')
    buffer.add(state, action, reward, next_state, done)
    states, actions, rewards, next_states, dones = buffer.sample()
    print(len(buffer))
    ```
    """
    def __init__(self, action_size, buffer_size, batch_size, seed, device='cpu'):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of the replay buffer."""
        return len(self.memory)
