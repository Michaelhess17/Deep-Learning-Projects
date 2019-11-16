import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from collections import deque
import random
import tqdm
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import numpy as np
import cv2
import torchvision
import gym
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# env = gym_super_mario_bros.make('SuperMarioBros-1-2-v1')
# env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = gym.make('CartPole-v1')
device = torch.device('cuda')
inx = env.observation_space.shape[0]
print(inx)
DISCOUNT = 0.98
REPLAY_MEMORY_SIZE = 500000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 200  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = 'CartPole'
MIN_REWARD = 4  # For model save
MEMORY_FRACTION = 0.20
EPISODES = 200_000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.9975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
AGGREGATE_STEPS_EVERY = 50  # episodes
SHOW_PREVIEW = True

# Exploration settings
EPSILON = 1  # not a constant, going to be decayed


class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(inx, 128)
        # linear layer (500 -> 10)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, env.action_space.n)
        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.dropout(self.fc4(x)))
        return x


class DQNAgent:
    def __init__(self):
        # gets trained every step
        self.policy_model = QNetwork().to(device)
        self.target_net = QNetwork().to(device)
        self.target_net.load_state_dict(self.policy_model.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_model.parameters(), lr=1e-3)

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.target_update_counter = 0
        self.loss = None


    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)


    def get_qs(self, state):
        return self.target_net.forward(state)


    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        current_states = np.array([transition[0].cpu().numpy() for transition in minibatch])
        # current_states = current_states.reshape(-1, 1, inx, iny)
        current_states = torch.tensor(current_states, dtype=torch.float32).to(device)
        current_qs_list = self.target_net.forward(current_states)
        new_current_states = np.array([transition[3].cpu().numpy() for transition in minibatch])
        # new_current_states = new_current_states.reshape(-1, 1, inx, iny)
        new_current_states = torch.tensor(new_current_states, dtype=torch.float32).to(device)
        future_qs_list = self.target_net.forward(new_current_states)
        X = []
        y = []

        for index, (state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = torch.max(future_qs_list.clone().detach()[index])
                new_q = torch.tensor(reward).to(device) + DISCOUNT * max_future_q
            else:
                new_q = torch.tensor(reward)
            current_qs = current_qs_list[index]
            current_qs[action] = new_q.to(device)
            X.append(state.cpu().numpy())
            y.append(current_qs.detach().cpu().numpy())
        y = np.array(y, dtype=np.float32).reshape(MINIBATCH_SIZE, env.action_space.n)
        X = torch.tensor(np.array(X).reshape(MINIBATCH_SIZE, inx), dtype=torch.float32).to(device)
        y = torch.from_numpy(y).to(device)

        predicted_value = self.policy_model.forward(X)
        self.loss = F.mse_loss(predicted_value, y)
        self.optimizer.zero_grad()
        self.loss.backward()

        for param in self.policy_model.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_net.load_state_dict(self.policy_model.state_dict())
        self.target_update_counter = 0


agent = DQNAgent()
ep_rewards = []
for episode in tqdm.tqdm(range(1, EPISODES + 1), ascii=True, unit="episode"):
    episode_reward = 0
    step = 1
    current_state = env.reset()
    current_state = torch.tensor(current_state).to(device)
    done = False
    current_max_fitness = 0
    fitness_current = 0
    counter = 0
    counter_2 = 0
    while not done:
        if np.random.random() > EPSILON:
            action = np.argmax(agent.get_qs(current_state))
        else:
            action = np.random.randint(0, env.action_space.n)
        new_state, reward, done, info = env.step(action)
        new_state = torch.tensor(new_state).to(device)
        episode_reward += reward
        env.render()
        if episode_reward > current_max_fitness:
            current_max_fitness = float(episode_reward)
            counter = 0
        else:
            counter += 1
        if done or counter == 250:
            episode_reward -= 2
            # plot_durations(ep_rewards)
            done = True
        elif not counter_2 % 7:
                agent.train(done, step)
        agent.update_replay_memory((current_state, action, reward, new_state, done))

        current_state = new_state
        step += 1
        counter_2 += 1
    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    agent.train(done, step)
    print(episode_reward)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        print(f'max: {max_reward} // avg: {average_reward}')

        # Save model, but only when min reward is greater or equal a set value
        if average_reward >= MIN_REWARD:
            torch.save(agent.target_net.state_dict(), f'mario_dqn_{episode}')
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

