import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os
from skimage import transform  # This requires scikit-image, for resizing images
from skimage.color import rgb2gray  # For converting images to grayscale

# Setup the environment
env = gym_super_mario_bros.make("SuperMarioBros-v0")
env = JoypadSpace(env, COMPLEX_MOVEMENT)


# Define the DQN model
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class Agent:
    def __init__(self, device="cpu", mode="test"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = DQN(84 * 84, env.action_space.n).to(self.device)
        self.target_model = DQN(84 * 84, env.action_space.n).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        if mode == "test":
            self.load()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return env.action_space.sample()
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_values = self.model(state)
        return np.argmax(action_values.cpu().data.numpy())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            action = torch.LongTensor([action]).to(self.device)
            reward = torch.FloatTensor([reward]).to(self.device)
            done = torch.FloatTensor([done]).to(self.device)

            current_q = self.model(state).gather(1, action.unsqueeze(1)).squeeze(1)
            next_q = self.target_model(next_state).detach().max(1)[0]
            target_q = reward + (self.gamma * next_q * (1 - done))

            # This creates a mask to zero-out all other Q-values except for the action taken
            q_update = self.model(state)
            q_update = q_update.gather(1, action.unsqueeze(1)).squeeze(1)

            loss = nn.MSELoss()(current_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

    def save(self, filename="./109061225_hw2_data"):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            filename,
        )

    def load(self, filename="./109061225_hw2_data"):
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.model.to(
            self.device
        )  # Ensure model is on the correct device after loading
