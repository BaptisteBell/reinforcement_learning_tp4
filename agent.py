from collections import deque
from model import *
import random
import typing as t
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy


class DeepQLearningAgent:
    def __init__(self,
                 learning_rate: float,
                 gamma: float,
                 n_actions: int,
                 input_dim : t.Tuple[int] = (4, 84, 84),
                 epsilon: float =1.0,
                 epsilon_min: float = 0.1
                 ):

        self.lr = learning_rate
        self.gamma = gamma
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.input_dim = input_dim
        self.memory = deque(maxlen=10000)
        self.epsilon_start = epsilon

        self.model, self.optimizer, self.loss = build_dqn(self.input_dim, self.n_actions, lr=self.lr)
        
        self.target_model = copy.deepcopy(self.model)
        self.target_update_frequency = 500
        self.learn_step_counter = 0
        self.frame_count = 0
        self.epsilon_decay_steps = 100000

    def add_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample_memory(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        return np.array(states), actions, rewards, np.array(next_states), dones
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

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

        actions = actions.view(-1, 1)

        q_values = self.model(states).gather(1, actions).squeeze()

        with torch.no_grad():
            max_next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        loss = self.loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()


        self.frame_count += 1
        self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * \
                    max(0, (self.epsilon_decay_steps - self.frame_count) / self.epsilon_decay_steps)

        #self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_frequency == 0:
            self.update_target_model()