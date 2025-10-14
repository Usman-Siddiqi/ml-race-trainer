"""
improved_visual_training.py

Improved visual training with better debugging and movement encouragement.
"""

import numpy as np
import torch
import pygame
import time
from race_environment import RaceCarEnv
from train_model import DQNAgent, action_to_continuous


def improved_visual_training(episodes=1000, render_interval=10):
    """
    Train with improved parameters and debugging.
    
    Args:
        episodes: Number of training episodes
        render_interval: Render every N episodes
    """
    print("Starting Improved Visual Training Mode")
    print("=" * 50)
    print("Fixed issues:")
    print("- Removed 'no action' from action space")
    print("- Increased epsilon minimum to 0.1 (more exploration)")
    print("- Better reward structure to encourage movement")
    print("- Added movement penalties for standing still")
    print("=" * 50)
    
    # Create environment with rendering
    env = RaceCarEnv(render_mode='human')
    
    # Create agent with improved parameters
    agent = DQNAgent(epsilon_min=0.1)  # More exploration
    
    # Training metrics
    scores = []
    best_score = -float('inf')
    stuck_episodes = 0
    
    try:
        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            movement_count = 0
            
            # Determine if we should render this episode
            should_render = episode % render_interval == 0
            
            print(f"Episode {episode + 1}/{episodes} {'(RENDERING)' if should_render else '(FAST)'}")
            
            while True:
                # Choose action
                action_idx = agent.act(state)
                action = action_to_continuous(action_idx)
                
                # Track movement
                if action[0] != 0 or action[1] != 0:
                    movement_count += 1
                
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
            
            # Check if episode was "stuck" (very little movement)
            movement_ratio = movement_count / steps if steps > 0 else 0
            if movement_ratio < 0.3:  # Less than 30% of steps had movement
                stuck_episodes += 1
                print(f"  WARNING: Episode {episode + 1} had low movement ({movement_ratio:.2%})")
            
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
                      f"Epsilon: {agent.epsilon:.3f}, Checkpoints: {info.get('checkpoints_passed', 0)}, "
                      f"Movement: {movement_ratio:.2%}")
    
    except KeyboardInterrupt:
        print("\nTraining stopped by user (Ctrl+C)")
    
    # Save final model
    torch.save(agent.q_network.state_dict(), 'final_race_model.pth')
    
    # Final statistics
    print(f"\nTraining Complete!")
    print(f"Episodes: {len(scores)}")
    print(f"Best Score: {best_score:.2f}")
    print(f"Final Average Score: {np.mean(scores[-10:]):.2f}")
    print(f"Stuck Episodes: {stuck_episodes} ({stuck_episodes/len(scores)*100:.1f}%)")
    
    if stuck_episodes > len(scores) * 0.3:
        print("\nWARNING: Many episodes had low movement. Consider:")
        print("- Increasing epsilon_min further")
        print("- Adjusting reward structure")
        print("- Adding more exploration")
    
    env.close()


def main():
    """Main function for improved visual training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Improved visual training mode for ML Race Trainer')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--render-interval', type=int, default=10,
                       help='Render every N episodes')
    
    args = parser.parse_args()
    
    improved_visual_training(episodes=args.episodes, render_interval=args.render_interval)


if __name__ == "__main__":
    main()
