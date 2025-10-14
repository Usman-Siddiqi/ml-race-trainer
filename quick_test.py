"""
quick_test.py

Quick test to verify the ML Race Trainer components work.
"""

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        import pygame
        print("✓ pygame imported")
    except ImportError as e:
        print(f"✗ pygame failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✓ numpy imported")
    except ImportError as e:
        print(f"✗ numpy failed: {e}")
        return False
    
    try:
        from race_environment import RaceCarEnv
        print("✓ race_environment imported")
    except ImportError as e:
        print(f"✗ race_environment failed: {e}")
        return False
    
    try:
        from train_model import DQNAgent
        print("✓ train_model imported")
    except ImportError as e:
        print(f"✗ train_model failed: {e}")
        return False
    
    return True

def test_environment():
    """Test that the environment can be created."""
    print("\nTesting environment creation...")
    
    try:
        from race_environment import RaceCarEnv
        env = RaceCarEnv(render_mode=None)
        print("✓ Environment created")
        
        obs, info = env.reset()
        print(f"✓ Environment reset, observation shape: {obs.shape}")
        
        action = [1, 0]  # Accelerate forward
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✓ Environment step successful, reward: {reward}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False

def test_simple_menu():
    """Test simple menu creation."""
    print("\nTesting simple menu...")
    
    try:
        import pygame
        pygame.init()
        screen = pygame.display.set_mode((400, 300))
        pygame.display.set_caption("Test")
        pygame.quit()
        print("✓ Simple pygame window created")
        return True
        
    except Exception as e:
        print(f"✗ Simple menu test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("ML Race Trainer - Quick Test")
    print("=" * 40)
    
    tests = [test_imports, test_environment, test_simple_menu]
    
    all_passed = True
    for test in tests:
        if not test():
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("✓ All tests passed!")
        print("\nTo run the application:")
        print("1. python simple_main.py")
        print("2. Use arrow keys to navigate")
        print("3. Press ENTER to select")
    else:
        print("✗ Some tests failed")
        print("\nTo fix issues:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Check that all files are in the same directory")

if __name__ == "__main__":
    main()
