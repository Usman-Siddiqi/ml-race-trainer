"""
test_generalization.py

Test whether a trained model can generalize to different tracks.
"""

import numpy as np
import torch
import pygame
import time
from race_environment import RaceCarEnv
from train_model import DQNAgent, action_to_continuous

class ModifiedRaceCarEnv(RaceCarEnv):
    """
    Modified race environment with a different track shape.
    """
    
    def _generate_track(self) -> dict:
        """
        Generate a different track shape (square instead of oval).
        """
        # Create a square track instead of oval
        center_x, center_y = self.track_width // 2, self.track_height // 2
        outer_size = 200
        inner_size = 100
        
        # Outer boundary (square)
        outer_points = [
            (center_x - outer_size, center_y - outer_size),
            (center_x + outer_size, center_y - outer_size),
            (center_x + outer_size, center_y + outer_size),
            (center_x - outer_size, center_y + outer_size)
        ]
        
        # Inner boundary (smaller square)
        inner_points = [
            (center_x - inner_size, center_y - inner_size),
            (center_x + inner_size, center_y - inner_size),
            (center_x + inner_size, center_y + inner_size),
            (center_x - inner_size, center_y + inner_size)
        ]
        
        return {'outer': outer_points, 'inner': inner_points}
    
    def _point_in_wall(self, x: float, y: float) -> bool:
        """
        Check if a point is inside a wall for the square track.
        """
        # Check if outside window bounds
        if x < 0 or x >= self.track_width or y < 0 or y >= self.track_height:
            return True
            
        # Check if outside outer boundary or inside inner boundary (square)
        center_x, center_y = self.track_width // 2, self.track_height // 2
        outer_size = 200
        inner_size = 100
        
        # Check outer boundary
        if (abs(x - center_x) > outer_size or abs(y - center_y) > outer_size):
            return True
            
        # Check inner boundary
        if (abs(x - center_x) < inner_size and abs(y - center_y) < inner_size):
            return True
            
        return False

def cleanup_pygame():
    """
    Properly cleanup pygame resources to prevent freezing.
    """
    print("  Cleaning up pygame resources...")
    
    # Clear all events
    pygame.event.clear()
    
    # Force display update
    try:
        pygame.display.flip()
        time.sleep(0.1)  # Brief pause
    except:
        pass
    
    # Clear any remaining events
    pygame.event.pump()

def test_model_generalization(model_path='best_race_model.pth', episodes=5):
    """
    Test how well a trained model performs on a different track.
    
    Args:
        model_path: Path to the trained model
        episodes: Number of test episodes
    """
    print("Testing Model Generalization")
    print("=" * 50)
    
    # Test on original track first
    print("Testing on ORIGINAL track (oval)...")
    try:
        original_env = RaceCarEnv(render_mode='human')
        results_original = test_model_on_track(original_env, model_path, episodes, "Original Oval Track")
        
        # Proper cleanup
        print("  Closing original environment...")
        original_env.close()
        del original_env
        
        # Clean up pygame
        cleanup_pygame()
        
        print("  Original track testing completed successfully!")
        
    except Exception as e:
        print(f"Error testing original track: {e}")
        return
    
    # Wait a moment between tests
    print("\nWaiting before next test...")
    time.sleep(1)
    
    print("\n" + "=" * 50)
    
    # Test on modified track
    print("Testing on MODIFIED track (square)...")
    try:
        modified_env = ModifiedRaceCarEnv(render_mode='human')
        results_modified = test_model_on_track(modified_env, model_path, episodes, "Modified Square Track")
        
        # Proper cleanup
        print("  Closing modified environment...")
        modified_env.close()
        del modified_env
        
        # Clean up pygame
        cleanup_pygame()
        
        print("  Modified track testing completed successfully!")
        
    except Exception as e:
        print(f"Error testing modified track: {e}")
        return
    
    print("\n" + "=" * 50)
    print("All tests completed successfully!")

def test_model_on_track(env, model_path, episodes, track_name):
    """
    Test a model on a specific track.
    """
    try:
        print(f"  Loading model from {model_path}...")
        
        # Create agent and load model
        agent = DQNAgent()
        agent.q_network.load_state_dict(torch.load(model_path))
        agent.epsilon = 0  # No exploration during testing
        
        print(f"  Model loaded successfully!")
        print(f"  Starting {episodes} episodes on {track_name}...")
        
        results = []
        
        for episode in range(episodes):
            print(f"\n{track_name} - Episode {episode + 1}/{episodes}")
            
            try:
                state, _ = env.reset()
                total_reward = 0
                steps = 0
                crashed = False
                completed = False
                
                max_steps = 2000  # Prevent infinite loops
                
                while steps < max_steps:
                    # Handle pygame events to prevent freezing
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            print("  User requested quit")
                            return results
                    
                    # Choose action
                    action_idx = agent.act(state)
                    action = action_to_continuous(action_idx)
                    
                    # Take step
                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    state = next_state
                    total_reward += reward
                    steps += 1
                    
                    crashed = info.get('crashed', False)
                    checkpoints = info.get('checkpoints_passed', 0)
                    
                    if checkpoints >= 10:
                        completed = True
                    
                    # Render with error handling
                    try:
                        env.render()
                    except Exception as render_error:
                        print(f"  Render error: {render_error}")
                    
                    if done:
                        break
                
                results.append({
                    'episode': episode + 1,
                    'score': total_reward,
                    'steps': steps,
                    'checkpoints': checkpoints,
                    'crashed': crashed,
                    'completed': completed
                })
                
                print(f"  Score: {total_reward:.2f}, Steps: {steps}, "
                      f"Checkpoints: {checkpoints}/10, Crashed: {crashed}, Completed: {completed}")
                
            except Exception as episode_error:
                print(f"  Error in episode {episode + 1}: {episode_error}")
                continue
        
        # Print summary
        if results:
            avg_score = np.mean([r['score'] for r in results])
            completion_rate = sum(1 for r in results if r['completed']) / len(results)
            crash_rate = sum(1 for r in results if r['crashed']) / len(results)
            
            print(f"\n{track_name} Summary:")
            print(f"  Average Score: {avg_score:.2f}")
            print(f"  Completion Rate: {completion_rate:.1%}")
            print(f"  Crash Rate: {crash_rate:.1%}")
        
        return results
        
    except FileNotFoundError:
        print(f"Model file {model_path} not found!")
        return None
    except Exception as e:
        print(f"Error testing model: {e}")
        return None

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test model generalization to different tracks')
    parser.add_argument('--model', type=str, default='best_race_model.pth',
                       help='Path to the trained model')
    parser.add_argument('--episodes', type=int, default=3,
                       help='Number of test episodes per track')
    
    args = parser.parse_args()
    
    try:
        test_model_generalization(model_path=args.model, episodes=args.episodes)
    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        # Final cleanup
        try:
            pygame.quit()
        except:
            pass
        print("Program ended.")

if __name__ == "__main__":
    main()
