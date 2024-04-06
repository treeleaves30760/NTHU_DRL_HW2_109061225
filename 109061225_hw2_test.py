import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from gym.wrappers import FrameStack
from gym.spaces import Box
from torchvision import transforms as T
import time

# Setup the environment
env = gym_super_mario_bros.make("SuperMarioBros-v0")
env = JoypadSpace(env, COMPLEX_MOVEMENT)


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape, antialias=True), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0)
        return observation


class DQN(nn.Module):
    """mini CNN structure
    input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.online = self.__build_cnn(c, output_dim)

        self.target = self.__build_cnn(c, output_dim)
        self.target.load_state_dict(self.online.state_dict())

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model="online"):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)

    def __build_cnn(self, c, output_dim):
        return nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=128, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1568, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )


class Agent:
    def __init__(
        self, state_dim=(4, 84, 84), action_dim=env.action_space.n, mode="test"
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.mode = mode
        if mode == "test":
            self.device = "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.net = DQN(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        if mode == "test":
            self.exploration_rate = 0.1
            self.exploration_rate_decay = 1
            self.exploration_rate_min = 0.1
        else:
            self.exploration_rate = 1
            self.exploration_rate_decay = 0.99999975
            self.exploration_rate_min = 0.1

        self.curr_step = 0

        self.save_every = 5e5  # no. of experiences between saving Mario Net

        self.memory = deque(maxlen=1000000)
        self.batch_size = 32

        self.gamma = 0.9

        self.burnin = 1e4  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync

        # Preprocessing transforms
        self.transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((84, 84)),  # Resize to match DQN input
                T.Grayscale(num_output_channels=1),
                T.ToTensor(),
            ]
        )

        self.load("./109061225_hw2_data_109")

    def preprocess(self, observation):
        # Apply transforms to observation
        observation = self.transform(observation)
        observation = observation.repeat(1, 4, 1, 1)
        return observation

    def act(self, state):
        """Given a state, choose an epsilon-greedy action"""
        state = self.preprocess(state).squeeze(0).to(self.device)
        # EXPLORE
        p = np.random.rand()
        if p < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = (
                state[0].__array__().copy()
                if isinstance(state, tuple)
                else state.__array__().copy()
            )
            state = torch.tensor(state, device=self.device).float().unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        if self.mode != "test":
            self.exploration_rate *= self.exploration_rate_decay
            self.exploration_rate = max(
                self.exploration_rate_min, self.exploration_rate
            )
        # increment step
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """Add the experience to memory"""

        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x

        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])

        self.memory.append(
            (
                state,
                next_state,
                action,
                reward,
                done,
            )
        )

    def recall(self):
        """Sample experiences from memory"""
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)

    def save(self, path="./109061225_hw2_data"):
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            path,
        )
        print(f"DQN saved to {path} at step {self.curr_step}")

    def load(self, path="./109061225_hw2_data"):
        save_dict = torch.load(path, map_location=self.device)  # Add map_location here
        self.net.load_state_dict(save_dict["model"])
        self.net = self.net.to(self.device)
        if self.mode != "test":
            self.exploration_rate = save_dict["exploration_rate"]
        print(f"DQN loaded from {path} at step {self.curr_step}")

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")

    mario = Agent(state_dim=(4, 84, 84), action_dim=env.action_space.n)
    total_reward = 0
    for i in range(50):
        # Test the model
        state = env.reset()
        episode_cycle = 0
        episode_reward = 0
        while True:
            env.render()
            action = mario.act(state)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state
            episode_cycle += 1
            if done:
                break
        total_reward += episode_reward
        print(f"Finished after {episode_cycle} cycles, total reward: {episode_reward}")
    print(f"Average reward: {total_reward / 50}")
    env.close()
