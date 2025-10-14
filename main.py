"""
main.py

Author: Usman Siddiqi
Date: 2025-10-13

Main interface for the ML Race Trainer application.
Provides options to train models or watch trained models race.
"""

import os
import sys
import pygame
import subprocess
from typing import List, Optional


class MainMenu:
    """
    Main menu interface for the ML Race Trainer.
    """
    
    def __init__(self):
        pygame.init()
        self.width = 600
        self.height = 400
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("ML Race Trainer - Main Menu")
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
            "Train New Model (Advanced Visual)",
            "Watch Trained Model",
            "Manual Control (Test Track)",
            "Exit"
        ]
        
        # Available models
        self.available_models = self.get_available_models()
    
    def get_available_models(self) -> List[str]:
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
            print("Starting fast training...")
            self.start_training(visual=False)
        elif option == "Train New Model (Visual)":
            print("Starting visual training...")
            self.start_training(visual=True)
        elif option == "Train New Model (Advanced Visual)":
            print("Starting advanced visual training...")
            self.start_advanced_visual_training()
        elif option == "Watch Trained Model":
            print("Starting model watching...")
            self.start_watching()
        elif option == "Manual Control (Test Track)":
            print("Starting manual control...")
            self.start_manual_control()
        elif option == "Exit":
            print("Exiting...")
            self.running = False
    
    def start_training(self, visual=False):
        """Start the training process."""
        mode = "visual" if visual else "fast"
        print(f"Starting {mode} training...")
        self.screen.fill(self.bg_color)
        
        # Show training message
        training_text = self.title_font.render(f"Training Model ({mode.title()})...", True, self.title_color)
        training_rect = training_text.get_rect(center=(self.width // 2, self.height // 2 - 50))
        self.screen.blit(training_text, training_rect)
        
        if visual:
            info_text = self.info_font.render("You can watch the AI learn! Check console for progress.", True, self.text_color)
        else:
            info_text = self.info_font.render("Fast training mode. Check console for progress.", True, self.text_color)
        info_rect = info_text.get_rect(center=(self.width // 2, self.height // 2))
        self.screen.blit(info_text, info_rect)
        
        pygame.display.flip()
        
        try:
            # Run training script
            cmd = [sys.executable, "train_model.py", "--mode", "train", "--episodes", "1000"]
            if visual:
                cmd.extend(["--visual", "--render-interval", "25"])
            
            print(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("Training completed successfully")
            print(f"Output: {result.stdout}")
            if result.stderr:
                print(f"Errors: {result.stderr}")
            
            # Show completion message
            self.screen.fill(self.bg_color)
            complete_text = self.title_font.render("Training Complete!", True, self.title_color)
            complete_rect = complete_text.get_rect(center=(self.width // 2, self.height // 2 - 50))
            self.screen.blit(complete_text, complete_rect)
            
            info_text = self.info_font.render("Press any key to return to menu", True, self.text_color)
            info_rect = info_text.get_rect(center=(self.width // 2, self.height // 2))
            self.screen.blit(info_text, info_rect)
            
            pygame.display.flip()
            
            # Wait for key press
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                        waiting = False
                    elif event.type == pygame.KEYDOWN:
                        waiting = False
            
            # Refresh available models
            self.available_models = self.get_available_models()
            
        except subprocess.CalledProcessError as e:
            print(f"Training failed: {e}")
            # Show error message
            self.screen.fill(self.bg_color)
            error_text = self.title_font.render("Training Failed!", True, (255, 0, 0))
            error_rect = error_text.get_rect(center=(self.width // 2, self.height // 2))
            self.screen.blit(error_text, error_rect)
            pygame.display.flip()
            pygame.time.wait(2000)
    
    def start_advanced_visual_training(self):
        """Start advanced visual training with live plots."""
        print("Starting advanced visual training...")
        self.screen.fill(self.bg_color)
        
        # Show training message
        training_text = self.title_font.render("Advanced Visual Training", True, self.title_color)
        training_rect = training_text.get_rect(center=(self.width // 2, self.height // 2 - 50))
        self.screen.blit(training_text, training_rect)
        
        info_text = self.info_font.render("Live plots + real-time visualization!", True, self.text_color)
        info_rect = info_text.get_rect(center=(self.width // 2, self.height // 2))
        self.screen.blit(info_text, info_rect)
        
        pygame.display.flip()
        
        try:
            # Run advanced visual training script
            cmd = [sys.executable, "visual_train.py", "--episodes", "1000", "--render-interval", "5"]
            print(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("Advanced training completed successfully")
            print(f"Output: {result.stdout}")
            if result.stderr:
                print(f"Errors: {result.stderr}")
            
            # Show completion message
            self.screen.fill(self.bg_color)
            complete_text = self.title_font.render("Advanced Training Complete!", True, self.title_color)
            complete_rect = complete_text.get_rect(center=(self.width // 2, self.height // 2 - 50))
            self.screen.blit(complete_text, complete_rect)
            
            info_text = self.info_font.render("Press any key to return to menu", True, self.text_color)
            info_rect = info_text.get_rect(center=(self.width // 2, self.height // 2))
            self.screen.blit(info_text, info_rect)
            
            pygame.display.flip()
            
            # Wait for key press
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                        waiting = False
                    elif event.type == pygame.KEYDOWN:
                        waiting = False
            
            # Refresh available models
            self.available_models = self.get_available_models()
            
        except subprocess.CalledProcessError as e:
            print(f"Advanced training failed: {e}")
            # Show error message
            self.screen.fill(self.bg_color)
            error_text = self.title_font.render("Advanced Training Failed!", True, (255, 0, 0))
            error_rect = error_text.get_rect(center=(self.width // 2, self.height // 2))
            self.screen.blit(error_text, error_rect)
            pygame.display.flip()
            pygame.time.wait(2000)
    
    def start_watching(self):
        """Start watching a trained model."""
        if not self.available_models:
            # Show no models message
            self.screen.fill(self.bg_color)
            no_model_text = self.title_font.render("No Models Available!", True, (255, 0, 0))
            no_model_rect = no_model_text.get_rect(center=(self.width // 2, self.height // 2 - 50))
            self.screen.blit(no_model_text, no_model_rect)
            
            info_text = self.info_font.render("Train a model first!", True, self.text_color)
            info_rect = info_text.get_rect(center=(self.width // 2, self.height // 2))
            self.screen.blit(info_text, info_rect)
            
            pygame.display.flip()
            pygame.time.wait(2000)
            return
        
        # Use the best model if available, otherwise the first one
        model_path = 'best_race_model.pth' if 'best_race_model.pth' in self.available_models else self.available_models[0]
        
        print(f"Starting to watch model: {model_path}")
        
        try:
            # Run watch script
            cmd = [sys.executable, "watch_model.py", "--model", model_path, "--episodes", "1"]
            print(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("Watching completed successfully")
            print(f"Output: {result.stdout}")
            if result.stderr:
                print(f"Errors: {result.stderr}")
        except subprocess.CalledProcessError as e:
            print(f"Watching failed: {e}")
    
    def start_manual_control(self):
        """Start manual control mode."""
        print("Starting manual control...")
        
        try:
            # Run the original race environment with manual control
            cmd = [sys.executable, "race_environment.py"]
            print(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("Manual control completed successfully")
            print(f"Output: {result.stdout}")
            if result.stderr:
                print(f"Errors: {result.stderr}")
        except subprocess.CalledProcessError as e:
            print(f"Manual control failed: {e}")
    
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
    print("ML Race Trainer - Main Menu")
    print("=" * 40)
    
    # Check if required files exist
    required_files = ['race_environment.py', 'train_model.py', 'watch_model.py']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"Error: Missing required files: {missing_files}")
        print("Please ensure all files are in the same directory.")
        return
    
    # Check if PyTorch is available
    try:
        import torch
        print("PyTorch is available ✓")
    except ImportError:
        print("Warning: PyTorch not found. Training will not work.")
        print("Install with: pip install torch")
    
    # Check if matplotlib is available
    try:
        import matplotlib
        print("Matplotlib is available ✓")
    except ImportError:
        print("Warning: Matplotlib not found. Training plots will not work.")
        print("Install with: pip install matplotlib")
    
    print("\nStarting main menu...")
    
    # Create and run main menu
    menu = MainMenu()
    menu.run()


if __name__ == "__main__":
    main()
