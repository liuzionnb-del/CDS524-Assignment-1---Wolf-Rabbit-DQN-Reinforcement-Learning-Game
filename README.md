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
Episode 0: Reward=-33.40, Captures=2, Loss=8.9723, ε=0.429
Episode 1: Reward=-56.00, Captures=1, Loss=3.8983, ε=0.157
Episode 2: Reward=-33.00, Captures=0, Loss=2.9430, ε=0.058
Episode 3: Reward=16.00, Captures=2, Loss=2.7385, ε=0.021
Episode 4: Reward=40.80, Captures=3, Loss=3.4864, ε=0.010
Episode 5: Reward=-8.00, Captures=0, Loss=3.4302, ε=0.010
Episode 6: Reward=44.90, Captures=4, Loss=2.3851, ε=0.010
Episode 7: Reward=92.90, Captures=6, Loss=4.7503, ε=0.010
Episode 8: Reward=71.40, Captures=4, Loss=4.6339, ε=0.010
Episode 9: Reward=111.90, Captures=8, Loss=5.4516, ε=0.010
Episode 10: Reward=-10.80, Captures=0, Loss=6.1191, ε=0.010
Episode 11: Reward=73.90, Captures=5, Loss=5.6814, ε=0.010
Episode 12: Reward=-86.50, Captures=0, Loss=5.2491, ε=0.010
Episode 13: Reward=-18.60, Captures=1, Loss=5.3447, ε=0.010
Episode 14: Reward=-2.90, Captures=1, Loss=5.3201, ε=0.010
Episode 15: Reward=-9.50, Captures=0, Loss=4.0526, ε=0.010
Episode 16: Reward=-8.50, Captures=0, Loss=4.8445, ε=0.010
Episode 17: Reward=21.60, Captures=2, Loss=4.3964, ε=0.010
Episode 18: Reward=21.40, Captures=2, Loss=5.0059, ε=0.010
Episode 19: Reward=0.80, Captures=1, Loss=4.4526, ε=0.010
Episode 20: Reward=64.30, Captures=4, Loss=3.7589, ε=0.010
Episode 21: Reward=93.10, Captures=6, Loss=4.7180, ε=0.010
Episode 22: Reward=116.50, Captures=8, Loss=5.7955, ε=0.010
Episode 23: Reward=60.40, Captures=4, Loss=5.4330, ε=0.010
Episode 24: Reward=70.60, Captures=5, Loss=5.6648, ε=0.010
Episode 25: Reward=91.60, Captures=5, Loss=5.2966, ε=0.010
Episode 26: Reward=172.20, Captures=9, Loss=6.1021, ε=0.010
Episode 27: Reward=134.00, Captures=8, Loss=7.5140, ε=0.010
Episode 28: Reward=216.20, Captures=11, Loss=6.0284, ε=0.010
Episode 29: Reward=140.20, Captures=8, Loss=8.0236, ε=0.010
Episode 30: Reward=164.20, Captures=11, Loss=7.1765, ε=0.010
Episode 31: Reward=87.50, Captures=6, Loss=6.4138, ε=0.010
Episode 32: Reward=-25.20, Captures=1, Loss=5.9547, ε=0.010
Episode 33: Reward=36.10, Captures=3, Loss=7.6498, ε=0.010
Episode 34: Reward=142.20, Captures=9, Loss=6.7230, ε=0.010
Episode 35: Reward=104.80, Captures=6, Loss=6.8889, ε=0.010
Episode 36: Reward=227.80, Captures=12, Loss=7.7431, ε=0.010
Episode 37: Reward=172.60, Captures=9, Loss=6.6151, ε=0.010
Episode 38: Reward=-60.80, Captures=0, Loss=7.2110, ε=0.010
Episode 39: Reward=238.20, Captures=12, Loss=7.6344, ε=0.010
Episode 40: Reward=101.90, Captures=6, Loss=7.8578, ε=0.010
Episode 41: Reward=115.80, Captures=7, Loss=7.7374, ε=0.010
Episode 42: Reward=129.00, Captures=7, Loss=8.7470, ε=0.010
Episode 43: Reward=31.80, Captures=2, Loss=7.7458, ε=0.010
Episode 44: Reward=202.90, Captures=12, Loss=7.5748, ε=0.010
Episode 45: Reward=249.60, Captures=13, Loss=7.1136, ε=0.010
Episode 46: Reward=150.80, Captures=9, Loss=6.9200, ε=0.010
Episode 47: Reward=201.60, Captures=11, Loss=7.7225, ε=0.010
Episode 48: Reward=241.40, Captures=12, Loss=7.8429, ε=0.010
Episode 49: Reward=211.50, Captures=13, Loss=6.4737, ε=0.010
## requirement
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
