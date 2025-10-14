"""
watch_model.py

Author: Usman Siddiqi
Date: 2025-10-13

This script allows you to watch a trained AI model race around the track.
"""

import numpy as np
import torch
import pygame
import os
import sys
from race_environment import RaceCarEnv
from train_model import DQNAgent, action_to_continuous


class ModelWatcher:
    """
    Class to watch and analyze trained model performance.
    """
    
    def __init__(self, model_path='best_race_model.pth'):
        self.model_path = model_path
        self.env = None
        self.agent = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model."""
        if not os.path.exists(self.model_path):
            print(f"Model file {self.model_path} not found!")
            print("Available model files:")
            for file in os.listdir('.'):
                if file.endswith('.pth'):
                    print(f"  - {file}")
            return False
        
        # Create environment with rendering
        self.env = RaceCarEnv(render_mode='human')
        
        # Create agent and load model
        self.agent = DQNAgent()
        self.agent.q_network.load_state_dict(torch.load(self.model_path))
        self.agent.epsilon = 0  # No exploration during watching
        
        print(f"Model loaded: {self.model_path}")
        return True
    
    def watch_single_episode(self, max_steps=2000):
        """
        Watch a single episode of the AI racing.
        
        Args:
            max_steps: Maximum steps before stopping
        """
        if not self.agent or not self.env:
            print("Model not loaded properly!")
            return
        
        state, _ = self.env.reset()
        total_reward = 0
        steps = 0
        crashed = False
        completed = False
        
        print("Starting episode... Press ESC to stop, SPACE to pause")
        
        running = True
        paused = False
        
        while running and steps < max_steps:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_r:
                        # Reset episode
                        state, _ = self.env.reset()
                        total_reward = 0
                        steps = 0
                        crashed = False
                        completed = False
                        print("Episode reset!")
            
            if not paused:
                # Choose action
                action_idx = self.agent.act(state)
                action = action_to_continuous(action_idx)
                
                # Take step
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                state = next_state
                total_reward += reward
                steps += 1
                
                crashed = info.get('crashed', False)
                checkpoints = info.get('checkpoints_passed', 0)
                
                if checkpoints >= 10:
                    completed = True
                
                # Render
                self.env.render()
                
                if done:
                    break
            
            else:
                # Still render when paused
                self.env.render()
        
        # Episode results
        print(f"\nEpisode Results:")
        print(f"  Steps: {steps}")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Checkpoints: {info.get('checkpoints_passed', 0)}/10")
        print(f"  Crashed: {crashed}")
        print(f"  Completed: {completed}")
        
        return {
            'steps': steps,
            'reward': total_reward,
            'checkpoints': info.get('checkpoints_passed', 0),
            'crashed': crashed,
            'completed': completed
        }
    
    def watch_multiple_episodes(self, num_episodes=5):
        """
        Watch multiple episodes and collect statistics.
        
        Args:
            num_episodes: Number of episodes to watch
        """
        if not self.agent or not self.env:
            print("Model not loaded properly!")
            return
        
        results = []
        
        print(f"Watching {num_episodes} episodes...")
        print("Press ESC to stop, SPACE to pause, R to reset current episode")
        
        for episode in range(num_episodes):
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            result = self.watch_single_episode()
            results.append(result)
            
            # Wait for user input to continue
            if episode < num_episodes - 1:
                print("Press any key to continue to next episode...")
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.env.close()
                            return results
                        elif event.type == pygame.KEYDOWN:
                            waiting = False
        
        # Print summary statistics
        self.print_statistics(results)
        return results
    
    def print_statistics(self, results):
        """Print summary statistics from multiple episodes."""
        if not results:
            return
        
        total_episodes = len(results)
        completed_episodes = sum(1 for r in results if r['completed'])
        crashed_episodes = sum(1 for r in results if r['crashed'])
        avg_reward = np.mean([r['reward'] for r in results])
        avg_steps = np.mean([r['steps'] for r in results])
        avg_checkpoints = np.mean([r['checkpoints'] for r in results])
        
        print(f"\n=== SUMMARY STATISTICS ===")
        print(f"Total Episodes: {total_episodes}")
        print(f"Completed: {completed_episodes} ({completed_episodes/total_episodes*100:.1f}%)")
        print(f"Crashed: {crashed_episodes} ({crashed_episodes/total_episodes*100:.1f}%)")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Steps: {avg_steps:.1f}")
        print(f"Average Checkpoints: {avg_checkpoints:.1f}/10")
    
    def close(self):
        """Close the environment."""
        if self.env:
            self.env.close()


def main():
    """Main function to run the model watcher."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Watch trained AI race car model')
    parser.add_argument('--model', type=str, default='best_race_model.pth',
                       help='Path to the trained model')
    parser.add_argument('--episodes', type=int, default=1,
                       help='Number of episodes to watch')
    
    args = parser.parse_args()
    
    # Create watcher
    watcher = ModelWatcher(args.model)
    
    if not watcher.agent or not watcher.env:
        print("Failed to load model. Exiting.")
        return
    
    try:
        if args.episodes == 1:
            watcher.watch_single_episode()
        else:
            watcher.watch_multiple_episodes(args.episodes)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        watcher.close()


if __name__ == "__main__":
    main()
