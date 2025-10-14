# ML Race Trainer

A 2D top-view racetrack environment for training autonomous cars using reinforcement learning. The car has 3 forward-facing sensors and must navigate the track without hitting the walls.

## Features

- **Reinforcement Learning Environment**: Gymnasium-compatible environment for training AI agents
- **Deep Q-Network (DQN) Training**: Complete training pipeline with experience replay and target networks
- **Visual Training**: Watch your AI learn to race around the track
- **Model Watching**: Observe trained models race with detailed statistics
- **Manual Control**: Test the track manually with keyboard controls
- **Easy-to-Use Interface**: Main menu for easy navigation between training and watching modes

## Installation

1. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   python main.py
   ```

## Usage

### Main Menu
Run `python main.py` to access the main menu with the following options:

1. **Train New Model (Fast)**: Train a new AI model using Deep Q-Network (no visualization, fastest)
2. **Train New Model (Visual)**: Train with periodic visualization of the AI learning
3. **Train New Model (Advanced Visual)**: Train with live plots and real-time visualization
4. **Watch Trained Model**: Watch an existing trained model race
5. **Manual Control**: Test the track manually with keyboard controls
6. **Exit**: Close the application

### Training a Model

The training process uses Deep Q-Network (DQN) with the following features:
- **Experience Replay**: Stores and samples from past experiences
- **Target Network**: Stable learning with periodic target network updates
- **Epsilon-Greedy Exploration**: Balances exploration and exploitation
- **Automatic Model Saving**: Saves best and periodic models during training

**Training Modes**:
1. **Fast Mode**: No visualization, maximum training speed
2. **Visual Mode**: Periodic visualization of AI learning (every 25 episodes)
3. **Advanced Visual Mode**: Live plots + real-time visualization with performance metrics

**Training Parameters**:
- Episodes: 1000 (configurable)
- Action Space: 9 discrete actions (combinations of acceleration/steering)
- Observation Space: 7 features (3 sensor distances, speed, angle, checkpoint progress)
- Reward System: Speed bonus, checkpoint rewards, crash penalties

### Watching Trained Models

Once you have a trained model, you can watch it race with:
- **Real-time Visualization**: See the AI car navigate the track
- **Performance Statistics**: Track checkpoints, crashes, and completion rates
- **Interactive Controls**: Pause, reset, or stop episodes
- **Multiple Episodes**: Watch multiple runs for performance analysis

### Manual Control

Test the track manually using keyboard controls:
- **Arrow Keys**: 
  - UP: Accelerate
  - DOWN: Brake/Reverse
  - LEFT: Turn left
  - RIGHT: Turn right
- **ESC**: Exit

## File Structure

```
ml-race-trainer/
├── main.py                 # Main menu interface
├── race_environment.py     # Gymnasium racing environment
├── train_model.py          # DQN training script
├── watch_model.py          # Model watching script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Environment Details

### Car Properties
- **Size**: 20x12 pixels
- **Max Speed**: 8 units per step
- **Sensors**: 3 forward-facing distance sensors (left, center, right)
- **Starting Position**: Right side of track, facing down

### Track
- **Shape**: Oval track with inner and outer boundaries
- **Checkpoints**: 10 checkpoints around the track
- **Collision Detection**: Car fails if any corner hits a wall

### Reward System
- **Speed Reward**: +0.1 per unit of speed
- **Checkpoint Reward**: +100 per checkpoint passed
- **Survival Reward**: +1 per step on track
- **Crash Penalty**: -100 for hitting walls
- **Completion Bonus**: +500 for completing all checkpoints
- **Time Penalty**: -0.1 per step (encourages efficiency)

## Training Tips

1. **Start with Few Episodes**: Begin with 100-200 episodes to test your setup
2. **Monitor Progress**: Watch the training progress in the console
3. **Check Model Files**: Look for `best_race_model.pth` and other saved models
4. **Adjust Parameters**: Modify learning rate, epsilon decay, etc. in `train_model.py`

## Troubleshooting

### Common Issues

1. **"No trained models found"**: Train a model first using the "Train New Model" option
2. **Training is slow**: This is normal for the first few episodes. Performance improves over time
3. **Model crashes frequently**: The AI needs more training episodes to learn proper navigation
4. **PyTorch not found**: Install PyTorch with `pip install torch`

### Performance Tips

- **GPU Training**: Install CUDA-enabled PyTorch for faster training
- **Reduce Episodes**: Start with fewer episodes for testing
- **Adjust Network Size**: Modify hidden layer sizes in `DQN` class for different performance

## Advanced Usage

### Command Line Training
```bash
# Fast training (no visualization)
python train_model.py --mode train --episodes 2000

# Visual training (periodic visualization)
python train_model.py --mode train --episodes 2000 --visual --render-interval 25

# Advanced visual training (live plots + real-time visualization)
python visual_train.py --episodes 2000 --render-interval 10
```

### Command Line Watching
```bash
python watch_model.py --model best_race_model.pth --episodes 5
```

### Custom Parameters
Modify the `DQNAgent` class in `train_model.py` to adjust:
- Learning rate
- Epsilon decay
- Network architecture
- Replay buffer size

## Contributing

Feel free to modify and improve the code:
- Add new reward functions
- Implement different RL algorithms
- Create new track layouts
- Add more sensor types
- Improve the visualization

## License

This project is open source and available under the MIT License.
