"""
debug_main.py

Simple debug version to test the main menu functionality.
"""

import pygame
import sys
import os

def test_main_menu():
    """Test the main menu without subprocess calls."""
    pygame.init()
    width = 600
    height = 400
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("ML Race Trainer - Debug")
    clock = pygame.time.Clock()
    
    # Colors
    bg_color = (30, 30, 30)
    button_color = (70, 130, 180)
    button_hover_color = (100, 149, 237)
    text_color = (255, 255, 255)
    title_color = (255, 215, 0)
    
    # Fonts
    title_font = pygame.font.Font(None, 48)
    button_font = pygame.font.Font(None, 32)
    info_font = pygame.font.Font(None, 24)
    
    # Menu state
    running = True
    selected_option = 0
    menu_options = [
        "Train New Model (Fast)",
        "Train New Model (Visual)", 
        "Train New Model (Advanced Visual)",
        "Watch Trained Model",
        "Manual Control (Test Track)",
        "Exit"
    ]
    
    print("Debug Main Menu Started")
    print("Use UP/DOWN arrows to navigate, ENTER to select, ESC to exit")
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                print("Window closed")
            
            elif event.type == pygame.KEYDOWN:
                print(f"Key pressed: {event.key}")
                if event.key == pygame.K_UP:
                    selected_option = (selected_option - 1) % len(menu_options)
                    print(f"Selected option: {selected_option} - {menu_options[selected_option]}")
                elif event.key == pygame.K_DOWN:
                    selected_option = (selected_option + 1) % len(menu_options)
                    print(f"Selected option: {selected_option} - {menu_options[selected_option]}")
                elif event.key == pygame.K_RETURN:
                    option = menu_options[selected_option]
                    print(f"SELECTED: {option}")
                    if option == "Exit":
                        running = False
                    else:
                        print(f"Would start: {option}")
                elif event.key == pygame.K_ESCAPE:
                    running = False
                    print("ESC pressed - exiting")
        
        # Draw everything
        screen.fill(bg_color)
        
        # Draw title
        title_text = title_font.render("ML Race Trainer - DEBUG", True, title_color)
        title_rect = title_text.get_rect(center=(width // 2, 80))
        screen.blit(title_text, title_rect)
        
        # Draw menu options
        start_y = 150
        option_height = 50
        
        for i, option in enumerate(menu_options):
            button_rect = pygame.Rect(150, start_y + i * option_height, 300, 40)
            
            if i == selected_option:
                color = button_hover_color
            else:
                color = button_color
            
            pygame.draw.rect(screen, color, button_rect)
            pygame.draw.rect(screen, text_color, button_rect, 2)
            
            text = button_font.render(option, True, text_color)
            text_rect = text.get_rect(center=button_rect.center)
            screen.blit(text, text_rect)
        
        # Draw instructions
        instructions = [
            "Use UP/DOWN arrows to navigate",
            "Press ENTER to select",
            "Press ESC to exit"
        ]
        
        for i, instruction in enumerate(instructions):
            text = info_font.render(instruction, True, (200, 200, 200))
            text_rect = text.get_rect(center=(width // 2, height - 100 + i * 20))
            screen.blit(text, text_rect)
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
    print("Debug main menu closed")

if __name__ == "__main__":
    test_main_menu()
