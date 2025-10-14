"""
visual_train.py

Advanced visual training mode that shows real-time training progress
with live visualization of the AI learning to race.
"""

import numpy as np
import torch
import pygame
import matplotlib.pyplot as plt
from collections import deque
import time
from race_environment import RaceCarEnv
from train_model import DQNAgent, action_to_continuous


class VisualTrainer:
    """
    Visual trainer that shows real-time training progress.
    """
    
    def __init__(self, episodes=1000, render_interval=10):
        self.episodes = episodes
        self.render_interval = render_interval
        
        # Create environment with rendering
        self.env = RaceCarEnv(render_mode='human')
        
        # Create agent
        self.agent = DQNAgent()
        
        # Training metrics
        self.scores = []
        self.avg_scores = []
        self.episode_times = []
        
        # Performance tracking
        self.best_score = -float('inf')
        self.episodes_completed = 0
        self.start_time = time.time()
        
        # Setup matplotlib for live plotting
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 4))
        self.fig.suptitle('ML Race Trainer - Live Training Progress')
        
    def update_plots(self):
        """Update the live training plots."""
        if len(self.scores) < 2:
            return
            
        # Clear and update score plot
        self.ax1.clear()
        self.ax1.plot(self.scores, alpha=0.3, color='blue')
        if len(self.avg_scores) > 0:
            self.ax1.plot(self.avg_scores, color='red', linewidth=2)
        self.ax1.set_title('Training Scores')
        self.ax1.set_xlabel('Episode')
        self.ax1.set_ylabel('Score')
        self.ax1.grid(True)
        
        # Update performance metrics
        self.ax2.clear()
        metrics = ['Best Score', 'Avg Score', 'Episodes', 'Time (min)']
        values = [
            self.best_score,
            np.mean(self.avg_scores[-10:]) if len(self.avg_scores) > 0 else 0,
            self.episodes_completed,
            (time.time() - self.start_time) / 60
        ]
        
        bars = self.ax2.bar(metrics, values, color=['gold', 'lightblue', 'lightgreen', 'lightcoral'])
        self.ax2.set_title('Training Metrics')
        self.ax2.set_ylabel('Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            self.ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                         f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.pause(0.01)
    
    def train_episode(self, episode):
        """Train a single episode with optional rendering."""
        state, _ = self.env.reset()
        total_reward = 0
        steps = 0
        episode_start = time.time()
        
        # Render if this episode should be shown
        should_render = episode % self.render_interval == 0
        
        while True:
            # Choose action
            action_idx = self.agent.act(state)
            action = action_to_continuous(action_idx)
            
            # Take step
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store experience
            self.agent.remember(state, action_idx, reward, next_state, done)
            
            # Train agent
            self.agent.replay()
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # Render if needed
            if should_render:
                self.env.render()
                # Small delay to make it watchable
                time.sleep(0.01)
            
            if done:
                break
        
        episode_time = time.time() - episode_start
        self.episode_times.append(episode_time)
        
        return total_reward, info
    
    def train(self):
        """Main training loop with live visualization."""
        print("Starting Visual Training Mode")
        print("=" * 50)
        print("You can watch the AI learn in real-time!")
        print("Close the matplotlib window to stop training early.")
        print("=" * 50)
        
        try:
            for episode in range(self.episodes):
                # Train episode
                total_reward, info = self.train_episode(episode)
                
                # Update metrics
                self.scores.append(total_reward)
                self.episodes_completed = episode + 1
                
                # Calculate average score
                if len(self.scores) >= 100:
                    avg_score = np.mean(self.scores[-100:])
                else:
                    avg_score = np.mean(self.scores)
                self.avg_scores.append(avg_score)
                
                # Update best score
                if total_reward > self.best_score:
                    self.best_score = total_reward
                    # Save best model
                    torch.save(self.agent.q_network.state_dict(), 'best_race_model.pth')
                
                # Update target network
                if episode % 10 == 0:
                    self.agent.update_target_network()
                
                # Update plots
                if episode % 5 == 0:  # Update plots every 5 episodes
                    self.update_plots()
                
                # Print progress
                if episode % 25 == 0:
                    elapsed_time = (time.time() - self.start_time) / 60
                    print(f"Episode {episode:4d} | Score: {total_reward:6.1f} | "
                          f"Avg: {avg_score:6.1f} | Epsilon: {self.agent.epsilon:.3f} | "
                          f"Checkpoints: {info.get('checkpoints_passed', 0):2d} | "
                          f"Time: {elapsed_time:5.1f}min")
                
                # Check if matplotlib window is closed
                if not plt.get_fignums():
                    print("\nTraining stopped by user (matplotlib window closed)")
                    break
                
        except KeyboardInterrupt:
            print("\nTraining stopped by user (Ctrl+C)")
        
        # Save final model
        torch.save(self.agent.q_network.state_dict(), 'final_race_model.pth')
        
        # Final statistics
        total_time = (time.time() - self.start_time) / 60
        print(f"\nTraining Complete!")
        print(f"Episodes: {self.episodes_completed}")
        print(f"Best Score: {self.best_score:.2f}")
        print(f"Final Average Score: {self.avg_scores[-1]:.2f}")
        print(f"Total Time: {total_time:.1f} minutes")
        print(f"Average Time per Episode: {np.mean(self.episode_times):.2f} seconds")
        
        # Keep plots open
        print("\nClose the matplotlib window to exit.")
        plt.ioff()
        plt.show()
        
        self.env.close()


def main():
    """Main function for visual training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visual training mode for ML Race Trainer')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--render-interval', type=int, default=10,
                       help='Render every N episodes')
    
    args = parser.parse_args()
    
    trainer = VisualTrainer(episodes=args.episodes, render_interval=args.render_interval)
    trainer.train()


if __name__ == "__main__":
    main()
