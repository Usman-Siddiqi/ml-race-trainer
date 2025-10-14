"""
simple_main.py

Simplified main menu that directly calls functions instead of using subprocess.
"""

import pygame
import sys
import os
from race_environment import RaceCarEnv
from train_model import train_dqn, test_model
from watch_model import ModelWatcher


class SimpleMainMenu:
    """
    Simplified main menu that directly calls functions.
    """
    
    def __init__(self):
        pygame.init()
        self.width = 600
        self.height = 400
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("ML Race Trainer - Simple Menu")
        self.clock = pygame.time.Clock()
        
        # Colors
        self.bg_color = (30, 30, 30)
        self.button_color = (70, 130, 180)
        self.button_hover_color = (100, 149, 237)
        self.text_color = (255, 255, 255)
        self.title_color = (255, 215, 0)
        
        # Fonts
        self.title_font = pygame.font.Font(None, 48)
        self.button_font = pygame.font.Font(None, 32)
        self.info_font = pygame.font.Font(None, 24)
        
        # Menu state
        self.running = True
        self.selected_option = 0
        self.menu_options = [
            "Train New Model (Fast)",
            "Train New Model (Visual)",
            "Watch Trained Model",
            "Manual Control (Test Track)",
            "Exit"
        ]
        
        # Available models
        self.available_models = self.get_available_models()
    
    def get_available_models(self):
        """Get list of available trained models."""
        models = []
        for file in os.listdir('.'):
            if file.endswith('.pth'):
                models.append(file)
        return sorted(models)
    
    def draw_title(self):
        """Draw the application title."""
        title_text = self.title_font.render("ML Race Trainer", True, self.title_color)
        title_rect = title_text.get_rect(center=(self.width // 2, 80))
        self.screen.blit(title_text, title_rect)
        
        subtitle_text = self.info_font.render("AI Car Racing with Reinforcement Learning", True, self.text_color)
        subtitle_rect = subtitle_text.get_rect(center=(self.width // 2, 120))
        self.screen.blit(subtitle_text, subtitle_rect)
    
    def draw_menu_options(self):
        """Draw the main menu options."""
        start_y = 180
        option_height = 50
        
        for i, option in enumerate(self.menu_options):
            # Button background
            button_rect = pygame.Rect(150, start_y + i * option_height, 300, 40)
            
            if i == self.selected_option:
                color = self.button_hover_color
            else:
                color = self.button_color
            
            pygame.draw.rect(self.screen, color, button_rect)
            pygame.draw.rect(self.screen, self.text_color, button_rect, 2)
            
            # Button text
            text = self.button_font.render(option, True, self.text_color)
            text_rect = text.get_rect(center=button_rect.center)
            self.screen.blit(text, text_rect)
    
    def draw_model_info(self):
        """Draw information about available models."""
        if self.available_models:
            info_text = f"Available models: {len(self.available_models)}"
            if len(self.available_models) <= 3:
                model_list = ", ".join(self.available_models)
                info_text += f" ({model_list})"
        else:
            info_text = "No trained models found. Train a model first!"
        
        text = self.info_font.render(info_text, True, self.text_color)
        text_rect = text.get_rect(center=(self.width // 2, self.height - 30))
        self.screen.blit(text, text_rect)
    
    def draw_instructions(self):
        """Draw control instructions."""
        instructions = [
            "Use UP/DOWN arrows to navigate",
            "Press ENTER to select",
            "Press ESC to exit"
        ]
        
        for i, instruction in enumerate(instructions):
            text = self.info_font.render(instruction, True, (200, 200, 200))
            text_rect = text.get_rect(center=(self.width // 2, self.height - 100 + i * 20))
            self.screen.blit(text, text_rect)
    
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.selected_option = (self.selected_option - 1) % len(self.menu_options)
                elif event.key == pygame.K_DOWN:
                    self.selected_option = (self.selected_option + 1) % len(self.menu_options)
                elif event.key == pygame.K_RETURN:
                    self.handle_selection()
                elif event.key == pygame.K_ESCAPE:
                    self.running = False
    
    def handle_selection(self):
        """Handle menu selection."""
        option = self.menu_options[self.selected_option]
        print(f"Selection made: {option}")
        
        if option == "Train New Model (Fast)":
            self.start_training(visual=False)
        elif option == "Train New Model (Visual)":
            self.start_training(visual=True)
        elif option == "Watch Trained Model":
            self.start_watching()
        elif option == "Manual Control (Test Track)":
            self.start_manual_control()
        elif option == "Exit":
            self.running = False
    
    def start_training(self, visual=False):
        """Start the training process."""
        mode = "visual" if visual else "fast"
        print(f"Starting {mode} training...")
        
        # Close pygame window temporarily
        pygame.quit()
        
        try:
            if visual:
                # Use the improved visual training
                from improved_visual_training import improved_visual_training
                improved_visual_training(episodes=1000, render_interval=10)
            else:
                # Run fast training directly
                train_dqn(episodes=1000, render_mode=None)
            print("Training completed successfully!")
            
        except Exception as e:
            print(f"Training failed: {e}")
        
        # Reinitialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("ML Race Trainer - Simple Menu")
        
        # Refresh available models
        self.available_models = self.get_available_models()
    
    def start_watching(self):
        """Start watching a trained model."""
        if not self.available_models:
            print("No trained models found. Train a model first!")
            return
        
        # Use the best model if available, otherwise the first one
        model_path = 'best_race_model.pth' if 'best_race_model.pth' in self.available_models else self.available_models[0]
        
        print(f"Starting to watch model: {model_path}")
        
        # Close pygame window temporarily
        pygame.quit()
        
        try:
            # Run watching directly
            watcher = ModelWatcher(model_path)
            watcher.watch_single_episode()
            watcher.close()
            print("Watching completed successfully!")
            
        except Exception as e:
            print(f"Watching failed: {e}")
        
        # Reinitialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("ML Race Trainer - Simple Menu")
    
    def start_manual_control(self):
        """Start manual control mode."""
        print("Starting manual control...")
        
        # Close pygame window temporarily
        pygame.quit()
        
        try:
            # Run manual control directly
            env = RaceCarEnv(render_mode='human')
            obs, info = env.reset()
            
            running = True
            while running:
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                
                # Get keyboard input for manual control
                keys = pygame.key.get_pressed()
                action = [0, 0]
                
                if keys[pygame.K_UP]:
                    action[0] = 1  # Accelerate
                elif keys[pygame.K_DOWN]:
                    action[0] = -1  # Brake
                
                if keys[pygame.K_LEFT]:
                    action[1] = -1  # Turn left
                elif keys[pygame.K_RIGHT]:
                    action[1] = 1  # Turn right
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                env.render()
                
                if terminated or truncated:
                    print(f"Episode ended. Checkpoints: {info['checkpoints_passed']}, Crashed: {info['crashed']}")
                    obs, info = env.reset()
            
            env.close()
            print("Manual control completed successfully!")
            
        except Exception as e:
            print(f"Manual control failed: {e}")
        
        # Reinitialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("ML Race Trainer - Simple Menu")
    
    def run(self):
        """Run the main menu loop."""
        while self.running:
            self.handle_events()
            
            # Draw everything
            self.screen.fill(self.bg_color)
            self.draw_title()
            self.draw_menu_options()
            self.draw_model_info()
            self.draw_instructions()
            
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()


def main():
    """Main function."""
    print("ML Race Trainer - Simple Main Menu")
    print("=" * 40)
    
    # Check if required files exist
    required_files = ['race_environment.py', 'train_model.py', 'watch_model.py']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"Error: Missing required files: {missing_files}")
        return
    
    # Create and run main menu
    menu = SimpleMainMenu()
    menu.run()


if __name__ == "__main__":
    main()
