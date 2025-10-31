"""
race_environment.py

Author: Usman Siddiqi
Date: 2025-10-13

This script defines the 2D top-view racetrack environment for training
autonomous cars using reinforcement learning. The car has 3 forward-facing
sensors and must navigate the track without hitting the walls.
"""

import pygame
import numpy as np
import math
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, List, Optional


class RaceCarEnv(gym.Env):
    """
    2D top-view racetrack environment for training autonomous racing cars.
    
    The car has 3 distance sensors pointing forward at different angles
    and must complete the track without hitting walls.
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, render_mode: Optional[str] = None, track_width: int = 800, track_height: int = 600, track_type: str = 'oval'):
        """
        Initialize the racing environment.
        
        Args:
            render_mode: Rendering mode ('human' or 'rgb_array')
            track_width: Width of the track window
            track_height: Height of the track window
            track_type: Type of track to generate ('oval' or 's_track')
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.track_width = track_width
        self.track_height = track_height
        self.track_type = track_type
        
        # Car properties
        self.car_width = 20
        self.car_height = 12
        self.car_speed = 0
        self.car_angle = 0
        self.max_speed = 8
        self.acceleration = 0.3
        self.deceleration = 0.2
        self.turn_speed = 3
        
        # Sensor properties
        self.sensor_length = 200
        self.sensor_angles = [-45, 0, 45]  # Widened sensor angles for better peripheral vision
        
        # Track properties
        if self.track_type == 's_track':
            self.track_points = self._generate_s_track()
            self.start_position = (100, 100)
            self.start_angle = 0
        else:
            self.track_points = self._generate_track()
            self.start_position = (650, 300)
            self.start_angle = 90
        
        # Current state
        self.car_x = self.start_position[0]
        self.car_y = self.start_position[1]
        self.car_angle = self.start_angle
        self.car_speed = 0
        
        # Episode tracking
        self.episode_steps = 0
        self.max_episode_steps = 2000
        self.crashed = False
        self.total_checkpoints = 10
        
        # Fixed checkpoint tracking variables
        self.current_checkpoint = 0  # Which checkpoint to hit next (0-9)
        self.checkpoints_passed_this_lap = 0  # How many checkpoints passed in current lap
        self.total_checkpoints_passed = 0  # Total checkpoints ever passed
        self.laps_completed = 0  # Number of complete laps
        self.last_distance_to_checkpoint = float('inf')  # For progress tracking
        
        # Generate checkpoints along the track
        self.checkpoints = self._generate_checkpoints()
        
        # Action space: [acceleration, steering]
        # acceleration: -1 (brake) to 1 (accelerate)
        # steering: -1 (left) to 1 (right)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation space: [sensor_distances (3), speed, angle_sin, angle_cos, checkpoint_progress]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, -1, -1, 0], dtype=np.float32),
            high=np.array([self.sensor_length, self.sensor_length, self.sensor_length,
                           self.max_speed, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        # Pygame setup
        if self.render_mode == 'human':
            pygame.init()
            self.screen = pygame.display.set_mode((self.track_width, self.track_height))
            pygame.display.set_caption("ML Race Trainer")
            self.clock = pygame.time.Clock()
        else:
            self.screen = None
    
    def _generate_track(self) -> List[Tuple[float, float]]:
        """
        Generate a simple oval track with inner and outer boundaries.
        
        Returns:
            List of track boundary points
        """
        # Create an oval track
        center_x, center_y = self.track_width // 2, self.track_height // 2
        outer_width, outer_height = 350, 250
        inner_width, inner_height = 200, 150
        
        # Outer boundary (clockwise)
        outer_points = []
        for i in range(0, 360, 5):
            angle = math.radians(i)
            x = center_x + outer_width * math.cos(angle)
            y = center_y + outer_height * math.sin(angle)
            outer_points.append((x, y))
        
        # Inner boundary (counter-clockwise)
        inner_points = []
        for i in range(360, 0, -5):
            angle = math.radians(i)
            x = center_x + inner_width * math.cos(angle)
            y = center_y + inner_height * math.sin(angle)
            inner_points.append((x, y))
        
        return {'outer': outer_points, 'inner': inner_points}

    def _generate_s_track(self) -> dict:
        """
        Generate a more complex S-shaped track.
        """
        points = []
        # Define the centerline of the S-track
        centerline = [
            (100, 100), (200, 100), (300, 200), (400, 300), (500, 400),
            (600, 500), (700, 500)
        ]

        # Generate outer and inner boundaries from the centerline
        outer_points = []
        inner_points = []
        track_width = 80

        for i in range(len(centerline) - 1):
            p1 = centerline[i]
            p2 = centerline[i+1]

            angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])

            # Perpendicular angle
            perp_angle = angle + math.pi / 2

            # Outer points
            outer_points.append((p1[0] + track_width * math.cos(perp_angle), p1[1] + track_width * math.sin(perp_angle)))
            outer_points.append((p2[0] + track_width * math.cos(perp_angle), p2[1] + track_width * math.sin(perp_angle)))

            # Inner points
            inner_points.append((p1[0] - track_width * math.cos(perp_angle), p1[1] - track_width * math.sin(perp_angle)))
            inner_points.append((p2[0] - track_width * math.cos(perp_angle), p2[1] - track_width * math.sin(perp_angle)))

        return {'outer': outer_points, 'inner': inner_points}
    
    def _generate_checkpoints(self) -> List[Tuple[float, float]]:
        """
        Generate checkpoint positions along the optimal racing line.
        """
        if self.track_type == 's_track':
            return self._generate_s_checkpoints()

        center_x, center_y = self.track_width // 2, self.track_height // 2
        outer_a, outer_b = 350, 250
        inner_a, inner_b = 200, 150
        blend = 0.55
        a_r = inner_a * (1 - blend) + outer_a * blend
        b_r = inner_b * (1 - blend) + outer_b * blend
        sx, sy = self.start_position
        start_theta = math.atan2(sy - center_y, sx - center_x)
        checkpoints: List[Tuple[float, float]] = []
        n = int(self.total_checkpoints) if hasattr(self, 'total_checkpoints') else 10
        for i in range(n):
            theta = start_theta - (2 * math.pi * i) / n
            x = center_x + a_r * math.cos(theta)
            y = center_y + b_r * math.sin(theta)
            checkpoints.append((x, y))
        return checkpoints[::-1]

    def _generate_s_checkpoints(self) -> List[Tuple[float, float]]:
        """
        Generate checkpoints for the S-track.
        """
        return [
            (150, 100), (250, 150), (350, 250), (450, 350),
            (550, 450), (650, 500)
        ]
    
    def _get_sensor_distances(self) -> List[float]:
        """
        Calculate distances from car sensors to track walls.
        """
        if self.track_type == 's_track':
            return self._get_s_sensor_distances()

        center_x, center_y = self.track_width // 2, self.track_height // 2
        outer_a, outer_b = 350, 250  # Outer ellipse radii
        inner_a, inner_b = 200, 150  # Inner ellipse radii

        distances = []
        
        for sensor_angle in self.sensor_angles:
            total_angle = math.radians(self.car_angle + sensor_angle)
            dx, dy = math.cos(total_angle), math.sin(total_angle)

            # Ray origin (car position) relative to center
            ox, oy = self.car_x - center_x, self.car_y - center_y

            # --- Intersection with outer ellipse ---
            a, b = outer_a, outer_b
            A = (dx / a)**2 + (dy / b)**2
            B = 2 * ((ox * dx / a**2) + (oy * dy / b**2))
            C = (ox / a)**2 + (oy / b)**2 - 1

            discriminant = B**2 - 4 * A * C
            d_outer = self.sensor_length
            if discriminant >= 0:
                t1 = (-B - math.sqrt(discriminant)) / (2 * A)
                t2 = (-B + math.sqrt(discriminant)) / (2 * A)
                if t1 > 0:
                    d_outer = t1
                elif t2 > 0:
                    d_outer = t2

            # --- Intersection with inner ellipse ---
            a, b = inner_a, inner_b
            A = (dx / a)**2 + (dy / b)**2
            B = 2 * ((ox * dx / a**2) + (oy * dy / b**2))
            C = (ox / a)**2 + (oy / b)**2 - 1

            discriminant = B**2 - 4 * A * C
            d_inner = self.sensor_length
            if discriminant >= 0:
                t1 = (-B - math.sqrt(discriminant)) / (2 * A)
                if t1 > 0:
                    d_inner = t1

            # The actual distance is the minimum of the two intersections
            min_dist = min(d_outer, d_inner, self.sensor_length)
            distances.append(min_dist)
            
        return distances

    def _get_s_sensor_distances(self) -> List[float]:
        """
        Calculate sensor distances for the S-track.
        """
        distances = []
        for sensor_angle in self.sensor_angles:
            total_angle = math.radians(self.car_angle + sensor_angle)
            cos_a = math.cos(total_angle)
            sin_a = math.sin(total_angle)

            sensor_distance = self.sensor_length
            for distance in range(1, self.sensor_length):
                x = self.car_x + distance * cos_a
                y = self.car_y + distance * sin_a

                if self._point_in_s_wall(x, y):
                    sensor_distance = distance
                    break
            distances.append(sensor_distance)
        return distances

    def _point_in_wall(self, x: float, y: float) -> bool:
        if self.track_type == 's_track':
            return self._point_in_s_wall(x, y)
        """
        Check if a point is inside a wall (outside track boundaries).
        
        Args:
            x, y: Point coordinates
            
        Returns:
            True if point is in a wall
        """
        # Check if outside window bounds
        if not (0 <= x < self.track_width and 0 <= y < self.track_height):
            return True

        # Check if outside outer boundary or inside inner boundary
        center_x, center_y = self.track_width // 2, self.track_height // 2
        dx, dy = x - center_x, y - center_y

        # Check outer boundary (ellipse)
        if (dx / 350)**2 + (dy / 250)**2 > 1:
            return True

        # Check inner boundary (ellipse)
        if (dx / 200)**2 + (dy / 150)**2 < 1:
            return True

        return False

    def _point_in_s_wall(self, x: float, y: float) -> bool:
        """
        Check if a point is inside a wall for the S-track.
        """
        if not (0 <= x < self.track_width and 0 <= y < self.track_height):
            return True

        # Check if the point is outside the wide path defined by the centerline
        centerline = [
            (100, 100), (200, 100), (300, 200), (400, 300), (500, 400),
            (600, 500), (700, 500)
        ]
        track_width = 80

        min_dist = float('inf')
        for i in range(len(centerline) - 1):
            p1 = np.array(centerline[i])
            p2 = np.array(centerline[i+1])
            p = np.array([x, y])

            # Vector from p1 to p2
            line_vec = p2 - p1
            # Vector from p1 to the point p
            point_vec = p - p1

            # Project point_vec onto line_vec
            line_len = np.linalg.norm(line_vec)
            if line_len == 0:
                dist = np.linalg.norm(point_vec)
            else:
                line_unitvec = line_vec / line_len
                t = np.dot(point_vec, line_unitvec)

                if t < 0.0:
                    dist = np.linalg.norm(p - p1)
                elif t > line_len:
                    dist = np.linalg.norm(p - p2)
                else:
                    nearest = p1 + t * line_unitvec
                    dist = np.linalg.norm(p - nearest)

            if dist < min_dist:
                min_dist = dist

        return min_dist > track_width

    def _check_collision(self) -> bool:
        """
        Check if the car has collided with track walls.
        
        Returns:
            True if collision detected
        """
        # Check car corners for collision
        car_corners = self._get_car_corners()
        
        for corner_x, corner_y in car_corners:
            if self._point_in_wall(corner_x, corner_y):
                return True
        
        return False
    
    def _get_car_corners(self) -> List[Tuple[float, float]]:
        """
        Get the four corners of the car based on current position and angle.
        
        Returns:
            List of corner coordinates
        """
        # Car corners relative to center
        half_width = self.car_width / 2
        half_height = self.car_height / 2
        
        corners = [
            (-half_width, -half_height),
            (half_width, -half_height),
            (half_width, half_height),
            (-half_width, half_height)
        ]
        
        # Rotate and translate corners
        angle_rad = math.radians(self.car_angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        rotated_corners = []
        for cx, cy in corners:
            # Rotate
            rx = cx * cos_a - cy * sin_a
            ry = cx * sin_a + cy * cos_a
            # Translate
            rx += self.car_x
            ry += self.car_y
            rotated_corners.append((float(rx), float(ry)))
        
        return rotated_corners
    
    def _check_checkpoints(self) -> float:
        """
        Check if car has passed the NEXT checkpoint in sequence and return reward.
        
        Returns:
            Checkpoint reward
        """
        if self.current_checkpoint >= len(self.checkpoints):
            return 0  # Safety check
        
        # Get the NEXT checkpoint that needs to be hit
        cp_x, cp_y = self.checkpoints[self.current_checkpoint]
        distance = math.sqrt((self.car_x - cp_x) ** 2 + (self.car_y - cp_y) ** 2)
        
        reward = 0
        
        # Check if close enough to the NEXT checkpoint
        if distance < 45:  # Checkpoint hit radius
            # Checkpoint passed!
            reward = 500  # Large reward for hitting checkpoint in correct order
            
            print(f"✅ Checkpoint {self.current_checkpoint} passed! "
                  f"Distance: {distance:.1f}")
            
            # Move to next checkpoint
            self.current_checkpoint = (self.current_checkpoint + 1) % self.total_checkpoints
            self.checkpoints_passed_this_lap += 1
            self.total_checkpoints_passed += 1
            
            # Check if completed a lap
            if self.checkpoints_passed_this_lap >= self.total_checkpoints:
                self.laps_completed += 1
                self.checkpoints_passed_this_lap = 0  # Reset for next lap
                lap_reward = 2000  # Massive bonus for completing lap
                reward += lap_reward
                print(f"🏁 LAP {self.laps_completed} COMPLETED! Bonus: +{lap_reward}")
        
        # Small reward for getting closer to the next checkpoint
        if distance < self.last_distance_to_checkpoint:
            reward += 2  # Small progress reward
        elif distance > self.last_distance_to_checkpoint:
            reward -= 1  # Small penalty for moving away
        
        self.last_distance_to_checkpoint = distance
        
        return reward
    
    def _calculate_reward(self) -> float:
        """
        Calculate the reward for the current step.
        
        Returns:
            Step reward
        """
        reward = 0
        
        # Speed reward (encourage forward movement) - increased
        reward += self.car_speed * 0.5
        
        # Movement reward (encourage any movement, not just speed)
        if self.car_speed > 0.1:
            reward += 0.5  # Bonus for moving
        
        # Checkpoint reward (sequential checkpoints only)
        checkpoint_reward = self._check_checkpoints()
        reward += checkpoint_reward
        
        # Staying on track reward
        if not self.crashed:
            reward += 1
        
        # Reward for high speed
        if self.car_speed > 5:
            reward += self.car_speed * 0.2

        # Penalty for crashing
        if self.crashed:
            reward -= 500
        
        # Penalty for standing still (encourage movement)
        if self.car_speed < 0.1:
            reward -= 0.5
        
        # Small penalty for time (encourage efficiency)
        reward -= 0.1
        
        return reward
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment to initial state.
        
        Returns:
            Observation and info dict
        """
        super().reset(seed=seed)
        
        # Reset car state
        self.car_x = self.start_position[0]
        self.car_y = self.start_position[1]
        self.car_angle = self.start_angle
        self.car_speed = 0
        
        # Reset episode tracking
        self.episode_steps = 0
        self.crashed = False
        
        # Reset checkpoint tracking. Instead of always using index 0, pick
        # the checkpoint that the car is actually facing / approaching so
        # checkpoint counting matches the car's clockwise movement.
        self.current_checkpoint = 0
        self.checkpoints_passed_this_lap = 0
        self.total_checkpoints_passed = 0
        self.laps_completed = 0

        # Calculate initial distance to the chosen starting checkpoint.
        if self.checkpoints:
            # Determine car heading unit vector
            heading_rad = math.radians(self.car_angle)
            heading_vec = (math.cos(heading_rad), math.sin(heading_rad))

            # Find the checkpoint that is nearest and roughly in front of the car
            best_idx = None
            best_dist = float('inf')

            for i, (cp_x, cp_y) in enumerate(self.checkpoints):
                vx = cp_x - self.car_x
                vy = cp_y - self.car_y
                dist = math.hypot(vx, vy)

                # Project vector to checkpoint onto heading to see if it's ahead
                forward_proj = vx * heading_vec[0] + vy * heading_vec[1]

                # Prefer checkpoints that are ahead (positive projection). Use a
                # small tolerance to allow near-side checkpoints.
                ahead_bonus = 0 if forward_proj > 5 else 10000

                score = dist + ahead_bonus
                if score < best_dist:
                    best_dist = score
                    best_idx = i

            if best_idx is None:
                best_idx = 0

            self.current_checkpoint = best_idx
            cp_x, cp_y = self.checkpoints[self.current_checkpoint]
            self.last_distance_to_checkpoint = math.sqrt((self.car_x - cp_x) ** 2 + (self.car_y - cp_y) ** 2)

        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: [acceleration, steering] values in [-1, 1]
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.episode_steps += 1
        
        # Extract actions
        acceleration_input = action[0]
        steering_input = action[1]
        
        # Update car speed
        if acceleration_input > 0:
            self.car_speed += acceleration_input * self.acceleration
        else:
            self.car_speed += acceleration_input * self.deceleration
        
        # Clamp speed
        self.car_speed = max(0, min(self.car_speed, self.max_speed))
        
        # Update car angle (only if moving)
        if self.car_speed > 0.1:
            self.car_angle += steering_input * self.turn_speed * (self.car_speed / self.max_speed)
            self.car_angle = self.car_angle % 360
        
        # Update car position
        angle_rad = math.radians(self.car_angle)
        self.car_x += self.car_speed * math.cos(angle_rad)
        self.car_y += self.car_speed * math.sin(angle_rad)
        
        # Check for collision
        self.crashed = self._check_collision()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination conditions
        terminated = self.crashed
        truncated = self.episode_steps >= self.max_episode_steps
        
        # Bonus termination condition: multiple laps completed
        if self.laps_completed >= 3:  # Allow up to 3 laps
            terminated = True
            reward += 5000  # Huge bonus for multiple laps
            print(f"🎉 AMAZING! {self.laps_completed} laps completed!")
        
        observation = self._get_observation()
        info = {
            'checkpoints_passed': self.total_checkpoints_passed,
            'checkpoints_this_lap': self.checkpoints_passed_this_lap,
            'current_checkpoint': self.current_checkpoint,
            'laps_completed': self.laps_completed,
            'crashed': self.crashed,
            'speed': self.car_speed
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation state.
        
        Returns:
            Observation array
        """
        # Get sensor distances
        sensor_distances = self._get_sensor_distances()
        
        # Normalize sensor distances
        normalized_distances = [d / self.sensor_length for d in sensor_distances]
        
        # Car state
        normalized_speed = self.car_speed / self.max_speed
        angle_sin = math.sin(math.radians(self.car_angle))
        angle_cos = math.cos(math.radians(self.car_angle))
        
        # Checkpoint progress (current lap progress)
        checkpoint_progress = self.checkpoints_passed_this_lap / self.total_checkpoints
        
        observation = np.array([
            normalized_distances[0],  # Left sensor
            normalized_distances[1],  # Center sensor
            normalized_distances[2],  # Right sensor
            normalized_speed,         # Current speed
            angle_sin,               # Angle sine
            angle_cos,               # Angle cosine
            checkpoint_progress      # Progress through current lap
        ], dtype=np.float32)
        
        return observation
    
    def render(self):
        """
        Render the environment.
        """
        if self.render_mode is None:
            return
        
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.track_width, self.track_height))
            pygame.display.set_caption("ML Race Trainer")
            self.clock = pygame.time.Clock()
        
        # Clear screen
        self.screen.fill((50, 50, 50))  # Dark gray background
        
        # Draw track boundaries
        if self.track_type == 's_track':
            pygame.draw.lines(self.screen, (100, 100, 100), False, self.track_points['outer'], 3)
            pygame.draw.lines(self.screen, (100, 100, 100), False, self.track_points['inner'], 3)
        else:
            if len(self.track_points['outer']) > 2:
                pygame.draw.polygon(self.screen, (100, 100, 100), self.track_points['outer'], 3)
            if len(self.track_points['inner']) > 2:
                pygame.draw.polygon(self.screen, (100, 100, 100), self.track_points['inner'], 3)
        
        # Draw checkpoints with proper highlighting
        for i, (cp_x, cp_y) in enumerate(self.checkpoints):
            if i == self.current_checkpoint:
                # Next checkpoint - bright green and larger
                color = (0, 255, 0)
                radius = 25
                width = 4
            else:
                # Other checkpoints - dim white
                color = (150, 150, 150)
                radius = 15
                width = 2
            
            pygame.draw.circle(self.screen, color, (int(cp_x), int(cp_y)), radius, width)
            
            # Draw checkpoint number
            font = pygame.font.Font(None, 24)
            text = font.render(str(i), True, (255, 255, 255))
            text_rect = text.get_rect(center=(cp_x, cp_y))
            self.screen.blit(text, text_rect)
        
        # Draw car
        car_corners = self._get_car_corners()
        car_color = (255, 0, 0) if self.crashed else (0, 100, 255)
        pygame.draw.polygon(self.screen, car_color, car_corners)
        
        # Draw sensors
        sensor_distances = self._get_sensor_distances()
        for i, sensor_angle in enumerate(self.sensor_angles):
            total_angle = math.radians(self.car_angle + sensor_angle)
            end_x = self.car_x + sensor_distances[i] * math.cos(total_angle)
            end_y = self.car_y + sensor_distances[i] * math.sin(total_angle)
            
            color = (255, 100, 100) if sensor_distances[i] < 20 else (100, 255, 100)
            start_pos = (float(self.car_x), float(self.car_y))
            end_pos = (float(end_x), float(end_y))
            pygame.draw.line(self.screen, color, start_pos, end_pos, 2)
        
        # Draw enhanced info
        font = pygame.font.Font(None, 32)
        info_lines = [
            f"Speed: {self.car_speed:.1f}",
            f"Laps: {self.laps_completed}",
            f"This Lap: {self.checkpoints_passed_this_lap}/10",
            f"Next CP: {self.current_checkpoint}",
            f"Total CPs: {self.total_checkpoints_passed}",
            f"Steps: {self.episode_steps}"
        ]
        
        for i, line in enumerate(info_lines):
            text = font.render(line, True, (255, 255, 255))
            self.screen.blit(text, (10, 10 + i * 35))
        
        if self.render_mode == 'human':
            pygame.display.flip()
            self.clock.tick(60)
    
    def close(self):
        """
        Close the environment.
        """
        if self.screen is not None:
            pygame.quit()


# Test the environment
if __name__ == "__main__":
    env = RaceCarEnv(render_mode='human')
    
    print("Testing Race Car Environment...")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    
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
            print(f"Episode ended. Laps: {info['laps_completed']}, "
                  f"Total checkpoints: {info['checkpoints_passed']}, "
                  f"Crashed: {info['crashed']}")
            obs, info = env.reset()
    
    env.close()
