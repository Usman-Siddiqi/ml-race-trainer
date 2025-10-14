"""
visual_training.py

Real-time visual training that shows the AI learning step by step.
"""

import numpy as np
import torch
import pygame
import time
from race_environment import RaceCarEnv
from train_model import DQNAgent, action_to_continuous


def visual_training(episodes=100, render_interval=5):
    """
    Train with real-time visualization.
    
    Args:
        episodes: Number of training episodes
        render_interval: Render every N episodes
    """
    print("Starting Visual Training Mode")
    print("=" * 50)
    print("You can watch the AI learn in real-time!")
    print("Close the pygame window to stop training early.")
    print("=" * 50)
    
    # Create environment with rendering
    env = RaceCarEnv(render_mode='human')
    
    # Create agent
    agent = DQNAgent()
    
    # Training metrics
    scores = []
    best_score = -float('inf')
    
    try:
        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            
            # Determine if we should render this episode
            should_render = episode % render_interval == 0
            
            print(f"Episode {episode + 1}/{episodes} {'(RENDERING)' if should_render else '(FAST)'}")
            
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
                
                # Render if this episode should be shown
                if should_render:
                    env.render()
                    # Small delay to make it watchable
                    time.sleep(0.05)
                    
                    # Handle pygame events to prevent freezing
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            print("\nTraining stopped by user (window closed)")
                            env.close()
                            return
                
                if done:
                    break
            
            # Update metrics
            scores.append(total_reward)
            
            # Update best score
            if total_reward > best_score:
                best_score = total_reward
                # Save best model
                torch.save(agent.q_network.state_dict(), 'best_race_model.pth')
                print(f"  New best score: {best_score:.2f} - Model saved!")
            
            # Update target network
            if episode % 10 == 0:
                agent.update_target_network()
            
            # Print progress
            if episode % 10 == 0:
                avg_score = np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores)
                print(f"  Score: {total_reward:.2f}, Avg: {avg_score:.2f}, "
                      f"Epsilon: {agent.epsilon:.3f}, Checkpoints: {info.get('checkpoints_passed', 0)}")
    
    except KeyboardInterrupt:
        print("\nTraining stopped by user (Ctrl+C)")
    
    # Save final model
    torch.save(agent.q_network.state_dict(), 'final_race_model.pth')
    
    # Final statistics
    print(f"\nTraining Complete!")
    print(f"Episodes: {len(scores)}")
    print(f"Best Score: {best_score:.2f}")
    print(f"Final Average Score: {np.mean(scores[-10:]):.2f}")
    
    env.close()


def main():
    """Main function for visual training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visual training mode for ML Race Trainer')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of training episodes')
    parser.add_argument('--render-interval', type=int, default=5,
                       help='Render every N episodes')
    
    args = parser.parse_args()
    
    visual_training(episodes=args.episodes, render_interval=args.render_interval)


if __name__ == "__main__":
    main()
