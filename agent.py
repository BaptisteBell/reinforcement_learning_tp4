from collections import deque
from model import *
import random
import typing as t
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class DeepQLearningAgent:
    def __init__(self,
                 learning_rate: float,
                 gamma: float,
                 n_actions: int,
                 input_dim : t.Tuple[int] = (4, 84, 84),
                 epsilon: float =1.0,
                 epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.995):

        self.lr = learning_rate
        self.gamma = gamma
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.input_dim = input_dim
        self.memory = deque(maxlen=100000)
        self.timestep = 0
        self.epsilon_start = epsilon

        self.model, self.optimizer, self.loss = build_dqn(input_dim, n_actions, lr=learning_rate) 

    def add_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample_memory(self, batch_size):
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        return np.array(states), actions, rewards, np.array(next_states), dones

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.n_actions)
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float)
            state = state.unsqueeze(0)
            state = state.to(self.model.device)
            q_values = self.model(state)
            return torch.argmax(q_values).item()

    def learn(self, batch):
        if len(self.memory) < batch:
            return
        
        states, actions, rewards, next_states, dones = self.sample_memory(batch)

        states = torch.tensor(states, dtype=torch.float).to(self.model.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.model.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.model.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.model.device)
        dones = torch.tensor(dones, dtype=torch.float).to(self.model.device)

        q_values = self.model(states).gather(1, actions).squeeze()

        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        loss = self.loss_fn(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return None