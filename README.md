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

## Requirement
pygame==2.5.2
numpy==1.24.3
torch==2.1.0
matplotlib==3.7.2

## How to Run
python DNQ_game.py

## Demo Video
https://youtu.be/GyjD5MinQ2Y

## Results
<img width="600" height="500" alt="92995eef-ab1a-483f-b576-a10081a9ab3e" src="https://github.com/user-attachments/assets/8cc2cd35-a598-4638-adec-daff740b8b77" />
<img width="1200" height="700" alt="training_results" src="https://github.com/user-attachments/assets/337a6d27-fe32-4b76-ad2d-1db35a171301" />


