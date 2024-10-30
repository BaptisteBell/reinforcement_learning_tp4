import typing as t
import gymnasium
import numpy as np
from agent import DeepQLearningAgent
import random
from collections import deque
from model_utils import *
import ale_py

#Suppress Overwriting existing videos warning
import warnings
import pandas as pd
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.wrappers.rendering")

env = gymnasium.make("ALE/Pong-v5")

# Hyperparameters
gamma = 0.99
batch_size = 32
n_actions = env.action_space.n
epochs = 500

agent = DeepQLearningAgent(learning_rate=0.0001, gamma=gamma, n_actions=n_actions)

def train_pong():
    agent.model.train()
    for epoch in range(epochs):
        state = env.reset()
        stacked_frames = [preprocess(state)] * 4
        state, stacked_frames = stack_frames(stacked_frames, state, True)
        done = False
        total_reward = 0
        while not done:
            action = agent.choose_action(state)

            next_state, reward, done, _ = env.step(agent.choose_action(state))
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

            agent.add_memory(state, action, reward, next_state, done)

            agent.learn(batch_size)

            state = next_state
            total_reward += reward

        print(f"Epoch {epoch} - Total reward: {total_reward}")
    
    save_model(agent.model)

#train_pong()

def play_pong():
    model = load_model()
    model.eval()
    state = env.reset()
    stacked_frames = [preprocess(state)] * 4
    state, stacked_frames = stack_frames(stacked_frames, state, True)
    done = False
    total_reward = 0
    while not done:
        env.render()
        action = model.choose_action(state)

        next_state, reward, done, _ = env.step(action)
        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

        state = next_state
        total_reward += reward

    print(f"Total reward: {total_reward}")

env.close()

#play_pong()