import typing as t
import numpy as np
from agent import DeepQLearningAgent
import random
from collections import deque
from model_utils import *

import ale_py
import gymnasium as gym

#Suppress Overwriting existing videos warning
import warnings
import pandas as pd
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.wrappers.rendering")

env = gym.make("ALE/Pong-v5")

# Hyperparameters
gamma = 0.99
batch_size = 32
n_actions = env.action_space.n # 6
epochs = 500

agent = DeepQLearningAgent(learning_rate=0.001, gamma=gamma, n_actions=n_actions)

def train_pong():
    agent.model.train()
    for epoch in range(epochs):
        state, info = env.reset()
        stacked_frames = deque(maxlen=4)  # Use deque for efficiency
        state, stacked_frames = stack_frames(stacked_frames, state, True)

        done = False
        total_reward = 0
        while not done:
            action = agent.choose_action(state)

            next_state, reward, termiated, truncated, _ = env.step(action)
            done = termiated or truncated

            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

            agent.add_memory(state, action, reward, next_state, done)

            agent.learn(batch_size)

            state = next_state
            
            reward = np.clip(reward, -1, 1)
            total_reward += reward

        print(f"Epoch {epoch} - Total reward: {total_reward}")

        if epoch % 10 == 0:
            save_model(agent.model)
    
    save_model(agent.model)

train_pong()

# def play_pong():
#     model = load_model()
#     model.eval()
#     state = env.reset()
#     stacked_frames = [preprocess(state)] * 4
#     state, stacked_frames = stack_frames(stacked_frames, state, True)
#     done = False
#     total_reward = 0
#     while not done:
#         env.render()
#         action = model.choose_action(state)

#         next_state, reward, done, _ = env.step(action)
#         next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

#         state = next_state
#         total_reward += reward

#     print(f"Total reward: {total_reward}")

env.close()

#play_pong()