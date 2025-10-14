"""
test_generalization.py

Test whether a trained model can generalize to different tracks.
"""

import numpy as np
import torch
import pygame
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
    original_env = RaceCarEnv(render_mode='human')
    test_model_on_track(original_env, model_path, episodes, "Original Oval Track")
    original_env.close()
    
    print("\n" + "=" * 50)
    
    # Test on modified track
    print("Testing on MODIFIED track (square)...")
    modified_env = ModifiedRaceCarEnv(render_mode='human')
    test_model_on_track(modified_env, model_path, episodes, "Modified Square Track")
    modified_env.close()


def test_model_on_track(env, model_path, episodes, track_name):
    """
    Test a model on a specific track.
    """
    try:
        # Create agent and load model
        agent = DQNAgent()
        agent.q_network.load_state_dict(torch.load(model_path))
        agent.epsilon = 0  # No exploration during testing
        
        results = []
        
        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            crashed = False
            completed = False
            
            print(f"\n{track_name} - Episode {episode + 1}")
            
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
                
                crashed = info.get('crashed', False)
                checkpoints = info.get('checkpoints_passed', 0)
                
                if checkpoints >= 10:
                    completed = True
                
                # Render
                env.render()
                
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
        
        # Print summary
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


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test model generalization to different tracks')
    parser.add_argument('--model', type=str, default='best_race_model.pth',
                       help='Path to the trained model')
    parser.add_argument('--episodes', type=int, default=3,
                       help='Number of test episodes per track')
    
    args = parser.parse_args()
    
    test_model_generalization(model_path=args.model, episodes=args.episodes)


if __name__ == "__main__":
    main()
