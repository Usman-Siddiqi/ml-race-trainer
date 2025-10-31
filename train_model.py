"""
train_model.py

Author: Usman Siddiqi
Date: 2025-10-13

This script trains a Deep Q-Network (DQN) model to control the race car
in the racing environment using reinforcement learning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import matplotlib.pyplot as plt
import os
from race_environment import RaceCarEnv


class DQN(nn.Module):
    """
    Deep Q-Network for race car control.
    """
    
    def __init__(self, input_size=7, hidden_size=128, output_size=9):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class ReplayBuffer:
    """
    Experience replay buffer for DQN training.
    """
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN Agent for training the race car.
    """
    
    def __init__(self, state_size=7, action_size=9, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Neural networks
        self.q_network = DQN(state_size, 128, action_size)
        self.target_network = DQN(state_size, 128, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Experience replay
        self.memory = ReplayBuffer(10000)
        self.batch_size = 64
        
        # Update target network
        self.update_target_network()
    
    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)
    
    def act(self, state):
        """Choose action using epsilon-greedy policy."""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self):
        """Train the model on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def action_to_continuous(action_idx):
    """
    Convert discrete action index to continuous action.
    
    Args:
        action_idx: Discrete action index (0-8)
        
    Returns:
        Continuous action [acceleration, steering]
    """
    # Define action mappings (removed "no action" to prevent getting stuck)
    actions = [
        [0.5, 0],    # 0: Slow forward
        [1, 0],      # 1: Forward only
        [0.5, -0.5], # 2: Slow forward + slight left
        [0.5, 0.5],  # 3: Slow forward + slight right
        [1, -1],     # 4: Forward + left
        [1, 1],      # 5: Forward + right
        [0.3, -1],   # 6: Slow forward + sharp left
        [0.3, 1],    # 7: Slow forward + sharp right
        [0.8, 0]     # 8: Medium forward
    ]
    
    return np.array(actions[action_idx], dtype=np.float32)


def train_dqn(episodes=1000, save_interval=100, render_mode=None, render_interval=50, track_type='oval'):
    """
    Train the DQN model.
    
    Args:
        episodes: Number of training episodes
        save_interval: Save model every N episodes
        render_mode: 'human' to watch training, None for faster training
        render_interval: Render every N episodes (only if render_mode='human')
        track_type: Type of track to use ('oval' or 's_track')
    """
    # Create environment
    env = RaceCarEnv(render_mode=render_mode, track_type=track_type)
    
    # Create agent
    agent = DQNAgent()
    
    # Training metrics
    scores = []
    avg_scores = []
    best_score = -float('inf')
    
    print("Starting DQN training...")
    print(f"Training for {episodes} episodes")
    print(f"Model will be saved every {save_interval} episodes")
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # Choose action
            action_idx = agent.act(state)
            action = action_to_continuous(action_idx)
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store experience
            agent.remember(state, action_idx, reward, next_state, done)
            
            # Train agent
            agent.replay()
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        scores.append(total_reward)
        
        # Calculate average score
        if len(scores) >= 100:
            avg_score = np.mean(scores[-100:])
            avg_scores.append(avg_score)
        else:
            avg_scores.append(np.mean(scores))
        
        # Update target network every 10 episodes
        if episode % 10 == 0:
            agent.update_target_network()
        
        # Render episode if in visual mode
        if render_mode == 'human' and episode % render_interval == 0:
            # Render the current state
            env.render()
            # Add a small delay to make it visible
            import time
            time.sleep(0.1)
        
        # Print progress
        if episode % 50 == 0:
            print(f"Episode {episode}, Score: {total_reward:.2f}, "
                  f"Avg Score: {avg_scores[-1]:.2f}, Epsilon: {agent.epsilon:.3f}, "
                  f"Checkpoints: {info.get('checkpoints_passed', 0)}")
        
        # Save model
        if episode % save_interval == 0 and episode > 0:
            if total_reward > best_score:
                best_score = total_reward
                torch.save(agent.q_network.state_dict(), 'best_race_model.pth')
                print(f"New best model saved! Score: {best_score:.2f}")
            
            torch.save(agent.q_network.state_dict(), f'race_model_episode_{episode}.pth')
            print(f"Model saved at episode {episode}")
    
    # Save final model
    torch.save(agent.q_network.state_dict(), 'final_race_model.pth')
    print("Training completed! Final model saved.")
    
    # Plot training progress
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title('Training Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    plt.subplot(1, 2, 2)
    plt.plot(avg_scores)
    plt.title('Average Scores (100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Average Score')
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.show()
    
    return agent


def test_model(model_path='best_race_model.pth', episodes=5, track_type='oval'):
    """
    Test a trained model.
    
    Args:
        model_path: Path to the saved model
        episodes: Number of test episodes
        track_type: Type of track to use ('oval' or 's_track')
    """
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found!")
        return
    
    # Create environment with rendering
    env = RaceCarEnv(render_mode='human', track_type=track_type)
    
    # Create agent and load model
    agent = DQNAgent()
    agent.q_network.load_state_dict(torch.load(model_path))
    agent.epsilon = 0  # No exploration during testing
    
    print(f"Testing model: {model_path}")
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        print(f"\nEpisode {episode + 1}")
        
        while True:
            # Choose action
            action_idx = agent.act(state)
            action = action_to_continuous(action_idx)
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # Render
            env.render()
            
            if done:
                break
        
        print(f"Score: {total_reward:.2f}, Steps: {steps}, "
              f"Checkpoints: {info.get('checkpoints_passed', 0)}, "
              f"Crashed: {info.get('crashed', False)}")
    
    env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train or test DQN race car model')
    parser.add_argument('--mode', choices=['train', 'test'], default='train',
                       help='Mode: train or test')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--model', type=str, default='best_race_model.pth',
                       help='Model path for testing')
    parser.add_argument('--visual', action='store_true',
                       help='Enable visual training (slower but you can watch)')
    parser.add_argument('--render-interval', type=int, default=50,
                       help='Render every N episodes (only with --visual)')
    parser.add_argument('--track', type=str, default='s_track',
                        help='Track to use for training or testing')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        render_mode = 'human' if args.visual else None
        train_dqn(episodes=args.episodes, render_mode=render_mode, render_interval=args.render_interval, track_type=args.track)
    else:
        test_model(model_path=args.model, track_type=args.track)
