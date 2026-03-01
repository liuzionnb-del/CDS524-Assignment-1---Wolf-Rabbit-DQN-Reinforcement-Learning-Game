# CDS524-Assignment-1-Wolf-Rabbit-DQN-Game
A predator-prey game where a wolf agent learns to hunt rabbits using Deep Q-Network (DQN) in a 10×10 grid environment with traps and food.

## 🎮 Game Overview
- **Wolf (DQN agent)**: Learns to chase and capture rabbits
- **Rabbit**: Moves randomly, prefers food locations
- **Traps**: Red squares, give -10 penalty
- **Food**: Green dots, attract rabbits

## 🧠 Key Features
- DQN implementation with experience replay and target networks
- 10-dimensional state space (positions, distances, boundaries, obstacles)
- Hierarchical reward function with dense feedback
- Real-time Pygame visualization with training statistics

## 📊 Results
- Best performance: **640 reward, 32 captures** in one episode (Episode 70)
- Stable performance: 500-600 reward, 20-30 captures per episode
- 
## Requriment
pygame==2.5.2
numpy==1.24.3
torch==2.1.0
matplotlib==3.7.2

## How to Run
python DNQ_game.py

## Demo Video
https://youtu.be/GyjD5MinQ2Y

## Results
Best: 640 reward, 32 captures at episode 70
