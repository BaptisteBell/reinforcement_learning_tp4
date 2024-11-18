import typing as t
import gymnasium
import numpy as np
from agent import DeepQLearningAgent
import random
from collections import deque
from model_utils import *
import ale_py
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.wrappers.rendering")

env = gymnasium.make("ALE/Pong-v5")

gamma = 0.95
batch_size = 64
n_actions = env.action_space.n
epochs = 500

agent = DeepQLearningAgent(learning_rate=0.00005, gamma=gamma, n_actions=n_actions)

def train_pong():
    print("Starting training...")
    agent.model.train()
    rewards_per_epoch = []
    agent.timestep = 0

    for epoch in range(epochs):
        state, _ = env.reset()
        stacked_frames = [preprocess(state)] * 4
        state, stacked_frames = stack_frames(stacked_frames, state, is_new_episode=True)
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state, stacked_frames = stack_frames(stacked_frames, next_state, is_new_episode=False)
            agent.add_memory(state, action, reward, next_state, done)

            agent.learn(batch_size)

            state = next_state
            total_reward += reward


            agent.timestep += 1

            if agent.timestep % 1000 == 0:
                agent.update_target_network()

        rewards_per_epoch.append(total_reward)
        print(f"Epoch {epoch + 1}/{epochs} - Total Reward: {total_reward}")

    print("Training completed.")
    save_model(agent.model)
    print("Model saved.")

def play_pong():
    print("Loading model for evaluation...")
    model = load_model()
    model.eval()
    total_rewards = []

    for episode in range(10):
        state, _ = env.reset()
        stacked_frames = [preprocess(state)] * 4
        state, stacked_frames = stack_frames(stacked_frames, state, is_new_episode=True)
        done = False
        total_reward = 0

        while not done:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(model.device)
                q_values = model(state_tensor)
                action = torch.argmax(q_values).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_state, stacked_frames = stack_frames(stacked_frames, next_state, is_new_episode=False)
            state = next_state
            total_reward += reward

        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    print(f"Average Reward over 10 Episodes: {np.mean(total_rewards)}")

# Run training and evaluation
if __name__ == "__main__":
    train_pong()
    play_pong()
    env.close()