import pygame
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import os
import json
import time
from datetime import datetime

# ==================== CONFIGURATION ====================
class Config:
    """Game and training configuration"""
    # Game settings
    GRID_SIZE = 10
    CELL_SIZE = 60
    INFO_PANEL_WIDTH = 300
    
    # Window size
    WINDOW_WIDTH = GRID_SIZE * CELL_SIZE + INFO_PANEL_WIDTH
    WINDOW_HEIGHT = GRID_SIZE * CELL_SIZE + 100
    
    # Colors
    BACKGROUND = (240, 240, 240)
    GRID_COLOR = (200, 200, 200)
    WOLF_COLOR = (100, 100, 100)      # Predator (DQN agent)
    RABBIT_COLOR = (255, 200, 200)     # Prey
    TRAP_COLOR = (255, 100, 100)       # Trap (negative reward)
    FOOD_COLOR = (100, 255, 100)       # Food (positive for rabbit)
    TEXT_COLOR = (0, 0, 0)
    BUTTON_COLOR = (150, 150, 150)
    BUTTON_HOVER_COLOR = (180, 180, 180)
    
    # DQN Hyperparameters
    STATE_SIZE = 10
    ACTION_SIZE = 4
    LEARNING_RATE = 0.001
    GAMMA = 0.95
    EPSILON_START = 1.0
    EPSILON_MIN = 0.01
    EPSILON_DECAY = 0.995
    BATCH_SIZE = 32
    MEMORY_SIZE = 10000
    TARGET_UPDATE_FREQ = 100
    
    # Training settings
    MAX_STEPS_PER_EPISODE = 200
    NUM_EPISODES = 500
    SAVE_FREQUENCY = 50  # Save model every N episodes
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== GAME ENVIRONMENT ====================
class WolfRabbitEnv:
    """
    Game environment with wolf (predator) and rabbit (prey)
    Wolf is controlled by DQN, rabbit moves randomly
    """
    
    def __init__(self, grid_size=Config.GRID_SIZE):
        """Initialize the environment"""
        self.grid_size = grid_size
        self.reset()
    
    def reset(self):
        """Reset the environment for a new episode"""
        # Initialize wolf position (avoid edges to give more room)
        self.wolf_pos = [random.randint(2, self.grid_size-3), 
                        random.randint(2, self.grid_size-3)]
        
        # Initialize rabbit position (different from wolf)
        self.rabbit_pos = self._generate_random_position(
            exclude=[self.wolf_pos]
        )
        
        # Generate traps (3 traps)
        self.traps = self._generate_positions(
            count=3, 
            exclude=[self.wolf_pos, self.rabbit_pos]
        )
        
        # Generate food (2 food locations)
        self.foods = self._generate_positions(
            count=2,
            exclude=[self.wolf_pos, self.rabbit_pos] + self.traps
        )
        
        return self._get_state()
    
    def _generate_random_position(self, exclude=[]):
        """Generate a random position not in exclude list"""
        while True:
            pos = [random.randint(0, self.grid_size-1),
                  random.randint(0, self.grid_size-1)]
            if pos not in exclude:
                return pos
    
    def _generate_positions(self, count, exclude=[]):
        """Generate multiple unique positions"""
        positions = []
        while len(positions) < count:
            pos = [random.randint(0, self.grid_size-1),
                  random.randint(0, self.grid_size-1)]
            if pos not in exclude and pos not in positions:
                positions.append(pos)
        return positions
    
    def _get_state(self):
        """
        Get state representation for DQN
        State vector (10 dimensions):
        - 0-1: Wolf coordinates (normalized)
        - 2-3: Rabbit coordinates (normalized)
        - 4: Distance to nearest trap (normalized)
        - 5: Distance to nearest food (normalized)
        - 6: Wolf-rabbit Manhattan distance (normalized)
        - 7: At boundary? (0/1)
        - 8: Number of surrounding traps (normalized)
        - 9: Number of surrounding food (normalized)
        """
        state = []
        
        # 1. Wolf position (normalized)
        state.append(self.wolf_pos[0] / self.grid_size)
        state.append(self.wolf_pos[1] / self.grid_size)
        
        # 2. Rabbit position (normalized)
        state.append(self.rabbit_pos[0] / self.grid_size)
        state.append(self.rabbit_pos[1] / self.grid_size)
        
        # 3. Distance to nearest trap
        trap_dist = self._get_min_distance(self.wolf_pos, self.traps)
        state.append(trap_dist / self.grid_size)
        
        # 4. Distance to nearest food
        food_dist = self._get_min_distance(self.wolf_pos, self.foods)
        state.append(food_dist / self.grid_size)
        
        # 5. Wolf-rabbit Manhattan distance
        distance = abs(self.wolf_pos[0] - self.rabbit_pos[0]) + \
                  abs(self.wolf_pos[1] - self.rabbit_pos[1])
        state.append(distance / (self.grid_size * 2))
        
        # 6. At boundary?
        at_boundary = int(self.wolf_pos[0] == 0 or self.wolf_pos[0] == self.grid_size-1 or
                         self.wolf_pos[1] == 0 or self.wolf_pos[1] == self.grid_size-1)
        state.append(at_boundary)
        
        # 7. Surrounding traps count
        surrounding_traps = self._count_surrounding(self.wolf_pos, self.traps)
        state.append(surrounding_traps / 4)
        
        # 8. Surrounding food count
        surrounding_food = self._count_surrounding(self.wolf_pos, self.foods)
        state.append(surrounding_food / 4)
        
        return np.array(state, dtype=np.float32)
    
    def _get_min_distance(self, pos, targets):
        """Get minimum Manhattan distance to any target"""
        if not targets:
            return self.grid_size
        distances = [abs(pos[0]-t[0]) + abs(pos[1]-t[1]) for t in targets]
        return min(distances)
    
    def _count_surrounding(self, pos, targets):
        """Count targets in adjacent cells"""
        count = 0
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
            check_pos = [pos[0]+dx, pos[1]+dy]
            if check_pos in targets:
                count += 1
        return count
    
    def step(self, action):
        """
        Execute one step in the environment
        Args:
            action: 0=up, 1=down, 2=left, 3=right
        Returns:
            next_state, reward, done, info
        """
        old_pos = self.wolf_pos.copy()
        old_distance = self._get_min_distance(self.wolf_pos, [self.rabbit_pos])
        
        # ===== 1. Move wolf based on action =====
        if action == 0:  # Up
            self.wolf_pos[1] = max(0, self.wolf_pos[1] - 1)
        elif action == 1:  # Down
            self.wolf_pos[1] = min(self.grid_size-1, self.wolf_pos[1] + 1)
        elif action == 2:  # Left
            self.wolf_pos[0] = max(0, self.wolf_pos[0] - 1)
        elif action == 3:  # Right
            self.wolf_pos[0] = min(self.grid_size-1, self.wolf_pos[0] + 1)
        
        # ===== 2. Calculate reward =====
        reward = -0.1  # Small penalty per step to encourage efficiency
        
        # Check capture
        captured = False
        if self.wolf_pos == self.rabbit_pos:
            reward += 20  # Big reward for catching rabbit
            captured = True
            # Respawn rabbit at new location
            self.rabbit_pos = self._generate_random_position(
                exclude=[self.wolf_pos] + self.traps + self.foods
            )
        
        # Check traps
        if self.wolf_pos in self.traps:
            reward -= 10  # Penalty for stepping on trap
            self.traps.remove(self.wolf_pos)  # Trap disappears
        
        # Distance-based reward (encourage getting closer to rabbit)
        new_distance = self._get_min_distance(self.wolf_pos, [self.rabbit_pos])
        if new_distance < old_distance:
            reward += 0.3  # Reward for moving closer
        else:
            reward -= 0.2  # Penalty for moving away
        
        # Boundary penalty
        if self.wolf_pos[0] in [0, self.grid_size-1] or \
           self.wolf_pos[1] in [0, self.grid_size-1]:
            reward -= 0.2
        
        # ===== 3. Move rabbit =====
        self._move_rabbit()
        
        # ===== 4. Check if episode should end =====
        done = False  # We don't end episodes, let DQN learn continuously
        
        # Additional info for visualization
        info = {
            'captured': captured,
            'trapped': self.wolf_pos in self.traps,
            'distance': new_distance
        }
        
        return self._get_state(), reward, done, info
    
    def _move_rabbit(self):
        """Move rabbit with simple AI"""
        # 30% chance to move each step
        if random.random() < 0.3:
            # Check if there's nearby food (attraction)
            nearby_food = []
            for food in self.foods:
                if abs(self.rabbit_pos[0] - food[0]) + \
                   abs(self.rabbit_pos[1] - food[1]) < 3:
                    nearby_food.append(food)
            
            if nearby_food and random.random() < 0.7:
                # Move towards food
                target = random.choice(nearby_food)
                dx = 1 if target[0] > self.rabbit_pos[0] else -1 if target[0] < self.rabbit_pos[0] else 0
                dy = 1 if target[1] > self.rabbit_pos[1] else -1 if target[1] < self.rabbit_pos[1] else 0
            else:
                # Random movement
                dx, dy = random.choice([(0,1), (1,0), (0,-1), (-1,0)])
            
            # Apply movement
            new_x = max(0, min(self.grid_size-1, self.rabbit_pos[0] + dx))
            new_y = max(0, min(self.grid_size-1, self.rabbit_pos[1] + dy))
            self.rabbit_pos = [new_x, new_y]


# ==================== DQN NEURAL NETWORK ====================
class DQN(nn.Module):
    """Deep Q-Network for wolf agent"""
    
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_size)
        
    def forward(self, x):
        """Forward pass through the network"""
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# ==================== DQN AGENT ====================
class DQNAgent:
    """DQN Agent that learns to control the wolf"""
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # DQN hyperparameters
        self.learning_rate = Config.LEARNING_RATE
        self.gamma = Config.GAMMA
        self.epsilon = Config.EPSILON_START
        self.epsilon_min = Config.EPSILON_MIN
        self.epsilon_decay = Config.EPSILON_DECAY
        self.batch_size = Config.BATCH_SIZE
        self.memory = deque(maxlen=Config.MEMORY_SIZE)
        self.update_target_freq = Config.TARGET_UPDATE_FREQ
        self.step_count = 0
        
        # Device (CPU/GPU)
        self.device = Config.DEVICE
        
        # Neural networks
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Initialize target network
        self._update_target()
        
        # ===== Training statistics =====
        self.losses = []           # Loss per training step
        self.episode_rewards = []   # Total reward per episode
        self.episode_captures = []  # Captures per episode
        self.episode_steps = []     # Steps per episode
        self.episode_losses = []    # Average loss per episode
        self.q_values = []          # Average Q-values
        self.epsilon_history = []    # Epsilon values
        self.action_counts = np.zeros(action_size)  # Action distribution
        
        # Detailed training data for analysis
        self.training_data = {
            'episode': [],
            'total_reward': [],
            'captures': [],
            'steps': [],
            'avg_loss': [],
            'avg_q_value': [],
            'epsilon': [],
            'timestamp': []
        }
    
    def _update_target(self):
        """Update target network with policy network weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Select action using epsilon-greedy policy"""
        # Exploration
        if random.random() <= self.epsilon:
            action = random.randrange(self.action_size)
        # Exploitation
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                action = np.argmax(q_values.cpu().data.numpy())
                # Record average Q-value
                self.q_values.append(q_values.mean().item())
        
        # Record action
        self.action_counts[action] += 1
        return action
    
    def replay(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return 0
        
        # Sample random batch from memory
        batch = random.sample(self.memory, self.batch_size)
        
        # Prepare batch data
        states = torch.FloatTensor([b[0] for b in batch]).to(self.device)
        actions = torch.LongTensor([b[1] for b in batch]).to(self.device)
        rewards = torch.FloatTensor([b[2] for b in batch]).to(self.device)
        next_states = torch.FloatTensor([b[3] for b in batch]).to(self.device)
        dones = torch.BoolTensor([b[4] for b in batch]).to(self.device)
        
        # Compute current Q-values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q-values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            next_q[dones] = 0.0
            target_q = rewards + self.gamma * next_q
        
        # Compute loss
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.update_target_freq == 0:
            self._update_target()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Record loss
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        return loss_value
    
    def save_model(self, filename):
        """Save model weights and training data"""
        # Save model
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, filename)
        
        print(f"💾 Model saved to {filename}")
    
    def load_model(self, filename):
        """Load model weights"""
        if os.path.exists(filename):
            checkpoint = torch.load(filename, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            print(f"📂 Model loaded from {filename}")
            return True
        return False
    
    def save_training_data(self, filename="training_data.json"):
        """Save all training statistics to JSON file"""
        # Prepare data for JSON
        data = {
            'episode_rewards': [float(x) for x in self.episode_rewards],
            'episode_captures': [int(x) for x in self.episode_captures],
            'episode_losses': [float(x) for x in self.episode_losses],
            'q_values': [float(x) for x in self.q_values],
            'epsilon_history': [float(x) for x in self.epsilon_history],
            'action_counts': [int(x) for x in self.action_counts],
            'hyperparameters': {
                'learning_rate': Config.LEARNING_RATE,
                'gamma': Config.GAMMA,
                'epsilon_start': Config.EPSILON_START,
                'epsilon_min': Config.EPSILON_MIN,
                'batch_size': Config.BATCH_SIZE,
                'memory_size': Config.MEMORY_SIZE,
                'state_size': Config.STATE_SIZE,
                'action_size': Config.ACTION_SIZE
            },
            'training_data': self.training_data
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"💾 Training data saved to {filename}")


# ==================== GAME VISUALIZATION ====================
class WolfRabbitGame:
    """Main game class with Pygame visualization"""
    
    def __init__(self):
        # Initialize Pygame
        pygame.init()
        self.width = Config.WINDOW_WIDTH
        self.height = Config.WINDOW_HEIGHT
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Wolf & Rabbit - DQN Reinforcement Learning")
        
        # Game components
        self.env = WolfRabbitEnv()
        self.agent = DQNAgent(Config.STATE_SIZE, Config.ACTION_SIZE)
        
        # UI components
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.large_font = pygame.font.Font(None, 32)
        
        # Training state
        self.current_episode = 0
        self.total_steps = 0
        self.running = True
        self.training = True
        self.paused = False
        
        # Buttons
        self.buttons = {
            'pause': pygame.Rect(Config.WINDOW_WIDTH - 280, 20, 80, 30),
            'save': pygame.Rect(Config.WINDOW_WIDTH - 190, 20, 80, 30),
            'plot': pygame.Rect(Config.WINDOW_WIDTH - 100, 20, 80, 30)
        }
    
    def draw_grid(self):
        """Draw the game grid"""
        # Draw grid lines
        for x in range(Config.GRID_SIZE + 1):
            pygame.draw.line(self.screen, Config.GRID_COLOR,
                           (x * Config.CELL_SIZE, 50),
                           (x * Config.CELL_SIZE, 
                            Config.GRID_SIZE * Config.CELL_SIZE + 50), 1)
        
        for y in range(Config.GRID_SIZE + 1):
            pygame.draw.line(self.screen, Config.GRID_COLOR,
                           (0, y * Config.CELL_SIZE + 50),
                           (Config.GRID_SIZE * Config.CELL_SIZE, 
                            y * Config.CELL_SIZE + 50), 1)
        
        # Draw coordinates
        for i in range(Config.GRID_SIZE):
            for j in range(Config.GRID_SIZE):
                if i == 0 or j == 0:
                    coord_text = self.small_font.render(f"{i},{j}", True, (150,150,150))
                    self.screen.blit(coord_text, 
                                   (i * Config.CELL_SIZE + 2, j * Config.CELL_SIZE + 52))
    
    def draw_agent(self, pos, color, size_scale=1.0, label=""):
        """Draw an agent (wolf or rabbit)"""
        x = pos[0] * Config.CELL_SIZE + Config.CELL_SIZE // 2
        y = pos[1] * Config.CELL_SIZE + 50 + Config.CELL_SIZE // 2
        radius = int(Config.CELL_SIZE // 3 * size_scale)
        
        # Draw circle
        pygame.draw.circle(self.screen, color, (x, y), radius)
        pygame.draw.circle(self.screen, (0, 0, 0), (x, y), radius, 2)
        
        # Draw label
        if label:
            text = self.small_font.render(label, True, (0, 0, 0))
            self.screen.blit(text, (x - 10, y - radius - 15))
    
    def draw_trap(self, pos):
        """Draw a trap"""
        x = pos[0] * Config.CELL_SIZE
        y = pos[1] * Config.CELL_SIZE + 50
        pygame.draw.rect(self.screen, Config.TRAP_COLOR, 
                        (x + 5, y + 5, Config.CELL_SIZE - 10, Config.CELL_SIZE - 10))
        # Draw X mark
        pygame.draw.line(self.screen, (0,0,0), (x + 8, y + 8), 
                        (x + Config.CELL_SIZE - 8, y + Config.CELL_SIZE - 8), 3)
        pygame.draw.line(self.screen, (0,0,0), (x + Config.CELL_SIZE - 8, y + 8),
                        (x + 8, y + Config.CELL_SIZE - 8), 3)
    
    def draw_food(self, pos):
        """Draw food"""
        x = pos[0] * Config.CELL_SIZE + Config.CELL_SIZE // 2
        y = pos[1] * Config.CELL_SIZE + 50 + Config.CELL_SIZE // 2
        pygame.draw.circle(self.screen, Config.FOOD_COLOR, (x, y), 8)
        pygame.draw.circle(self.screen, (0,0,0), (x, y), 8, 1)
    
    def draw_info_panel(self):
        """Draw information panel with training statistics"""
        panel_x = Config.GRID_SIZE * Config.CELL_SIZE + 20
        panel_y = 80
        
        # Title
        title = self.large_font.render("Training Info", True, Config.TEXT_COLOR)
        self.screen.blit(title, (panel_x, panel_y - 30))
        
        # Episode info
        episode_text = self.font.render(f"Episode: {self.current_episode}", 
                                       True, Config.TEXT_COLOR)
        self.screen.blit(episode_text, (panel_x, panel_y))
        
        step_text = self.font.render(f"Total Steps: {self.total_steps}", 
                                    True, Config.TEXT_COLOR)
        self.screen.blit(step_text, (panel_x, panel_y + 30))
        
        # DQN Parameters
        param_y = panel_y + 80
        params_title = self.font.render("DQN Parameters:", True, Config.TEXT_COLOR)
        self.screen.blit(params_title, (panel_x, param_y))
        
        epsilon_text = self.small_font.render(f"ε = {self.agent.epsilon:.3f}", 
                                            True, Config.TEXT_COLOR)
        self.screen.blit(epsilon_text, (panel_x + 10, param_y + 25))
        
        memory_text = self.small_font.render(f"Memory: {len(self.agent.memory)}", 
                                           True, Config.TEXT_COLOR)
        self.screen.blit(memory_text, (panel_x + 10, param_y + 45))
        
        # Statistics
        stats_y = param_y + 90
        stats_title = self.font.render("Statistics:", True, Config.TEXT_COLOR)
        self.screen.blit(stats_title, (panel_x, stats_y))
        
        if self.agent.episode_rewards:
            recent_rewards = self.agent.episode_rewards[-10:]
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            reward_text = self.small_font.render(f"Avg Reward: {avg_reward:.2f}", 
                                               True, Config.TEXT_COLOR)
            self.screen.blit(reward_text, (panel_x + 10, stats_y + 25))
        
        if self.agent.episode_captures:
            captures = self.agent.episode_captures[-1] if self.agent.episode_captures else 0
            capture_text = self.small_font.render(f"Captures: {captures}", 
                                                True, Config.TEXT_COLOR)
            self.screen.blit(capture_text, (panel_x + 10, stats_y + 45))
        
        # Action distribution
        action_y = stats_y + 90
        action_title = self.font.render("Actions:", True, Config.TEXT_COLOR)
        self.screen.blit(action_title, (panel_x, action_y))
        
        actions = ['Up', 'Down', 'Left', 'Right']
        for i, action in enumerate(actions):
            count = self.agent.action_counts[i]
            percent = (count / max(1, sum(self.agent.action_counts))) * 100
            action_text = self.small_font.render(f"{action}: {percent:.1f}%", 
                                               True, Config.TEXT_COLOR)
            self.screen.blit(action_text, (panel_x + 10, action_y + 25 + i * 18))
        
        # Legend
        legend_y = action_y + 120
        legend_title = self.font.render("Legend:", True, Config.TEXT_COLOR)
        self.screen.blit(legend_title, (panel_x, legend_y))
        
        # Wolf
        pygame.draw.circle(self.screen, Config.WOLF_COLOR, (panel_x + 10, legend_y + 30), 8)
        wolf_text = self.small_font.render("Wolf (DQN)", True, Config.TEXT_COLOR)
        self.screen.blit(wolf_text, (panel_x + 25, legend_y + 25))
        
        # Rabbit
        pygame.draw.circle(self.screen, Config.RABBIT_COLOR, (panel_x + 10, legend_y + 55), 8)
        rabbit_text = self.small_font.render("Rabbit", True, Config.TEXT_COLOR)
        self.screen.blit(rabbit_text, (panel_x + 25, legend_y + 50))
        
        # Trap
        pygame.draw.rect(self.screen, Config.TRAP_COLOR, (panel_x + 5, legend_y + 75, 16, 16))
        trap_text = self.small_font.render("Trap (-10)", True, Config.TEXT_COLOR)
        self.screen.blit(trap_text, (panel_x + 25, legend_y + 75))
        
        # Food
        pygame.draw.circle(self.screen, Config.FOOD_COLOR, (panel_x + 13, legend_y + 105), 8)
        food_text = self.small_font.render("Food", True, Config.TEXT_COLOR)
        self.screen.blit(food_text, (panel_x + 25, legend_y + 100))
    
    def draw_buttons(self):
        """Draw control buttons"""
        mouse_pos = pygame.mouse.get_pos()
        
        for name, rect in self.buttons.items():
            # Check hover
            color = Config.BUTTON_HOVER_COLOR if rect.collidepoint(mouse_pos) else Config.BUTTON_COLOR
            
            # Draw button
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, (0,0,0), rect, 2)
            
            # Draw text
            text = self.small_font.render(name.capitalize(), True, Config.TEXT_COLOR)
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)
    
    def handle_buttons(self, pos):
        """Handle button clicks"""
        for name, rect in self.buttons.items():
            if rect.collidepoint(pos):
                if name == 'pause':
                    self.paused = not self.paused
                    print(f"{'Paused' if self.paused else 'Resumed'} training")
                elif name == 'save':
                    self.agent.save_model(f"dqn_model_episode_{self.current_episode}.pth")
                    self.agent.save_training_data()
                elif name == 'plot':
                    self.plot_training_results()
    
    def plot_training_results(self):
        """Plot training progress"""
        if len(self.agent.episode_rewards) < 2:
            print("Not enough data to plot")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # 1. Episode Rewards
        axes[0,0].plot(self.agent.episode_rewards)
        axes[0,0].set_title('Episode Rewards')
        axes[0,0].set_xlabel('Episode')
        axes[0,0].set_ylabel('Total Reward')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Captures per Episode
        axes[0,1].plot(self.agent.episode_captures)
        axes[0,1].set_title('Captures per Episode')
        axes[0,1].set_xlabel('Episode')
        axes[0,1].set_ylabel('Captures')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Loss over time
        if self.agent.losses:
            # Smooth loss
            window = 100
            losses_smooth = np.convolve(self.agent.losses, 
                                       np.ones(window)/window, mode='valid')
            axes[0,2].plot(losses_smooth)
            axes[0,2].set_title('Training Loss (smoothed)')
            axes[0,2].set_xlabel('Training Step')
            axes[0,2].set_ylabel('Loss')
            axes[0,2].grid(True, alpha=0.3)
        
        # 4. Q-values
        if self.agent.q_values:
            axes[1,0].plot(self.agent.q_values)
            axes[1,0].set_title('Average Q-Values')
            axes[1,0].set_xlabel('Step')
            axes[1,0].set_ylabel('Q-Value')
            axes[1,0].grid(True, alpha=0.3)
        
        # 5. Epsilon Decay
        axes[1,1].plot(self.agent.epsilon_history)
        axes[1,1].set_title('Exploration Rate (ε)')
        axes[1,1].set_xlabel('Episode')
        axes[1,1].set_ylabel('Epsilon')
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Action Distribution
        actions = ['Up', 'Down', 'Left', 'Right']
        axes[1,2].bar(actions, self.agent.action_counts)
        axes[1,2].set_title('Action Distribution')
        axes[1,2].set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig('training_results.png')
        plt.show()
        print("📊 Training plots saved to 'training_results.png'")
    
    def run_episode(self):
        """Run a single training episode"""
        state = self.env.reset()
        episode_reward = 0
        episode_captures = 0
        episode_losses = []
        step = 0
        
        while step < Config.MAX_STEPS_PER_EPISODE:
            # Handle Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_buttons(event.pos)
            
            if not self.running or self.paused:
                # Still render when paused
                self.render()
                self.clock.tick(10)
                continue
            
            # Select action
            action = self.agent.act(state)
            
            # Execute action
            next_state, reward, done, info = self.env.step(action)
            
            # Store experience
            self.agent.remember(state, action, reward, next_state, done)
            
            # Train agent
            loss = self.agent.replay()
            if loss > 0:
                episode_losses.append(loss)
            
            # Update statistics
            episode_reward += reward
            if info['captured']:
                episode_captures += 1
            self.total_steps += 1
            
            # Update state
            state = next_state
            step += 1
            
            # Render
            self.render()
            
            # Control speed
            self.clock.tick(30)
        
        # Episode ended
        self.agent.episode_rewards.append(episode_reward)
        self.agent.episode_captures.append(episode_captures)
        self.agent.episode_steps.append(step)
        self.agent.epsilon_history.append(self.agent.epsilon)
        
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        self.agent.episode_losses.append(avg_loss)
        
        # Record detailed data
        self.agent.training_data['episode'].append(self.current_episode)
        self.agent.training_data['total_reward'].append(float(episode_reward))
        self.agent.training_data['captures'].append(int(episode_captures))
        self.agent.training_data['steps'].append(int(step))
        self.agent.training_data['avg_loss'].append(float(avg_loss))
        self.agent.training_data['avg_q_value'].append(
            float(np.mean(self.agent.q_values[-step:]) if self.agent.q_values else 0)
        )
        self.agent.training_data['epsilon'].append(float(self.agent.epsilon))
        self.agent.training_data['timestamp'].append(str(datetime.now()))
        
        # Print progress
        print(f"Episode {self.current_episode}: Reward={episode_reward:.2f}, "
              f"Captures={episode_captures}, Loss={avg_loss:.4f}, ε={self.agent.epsilon:.3f}")
        
        self.current_episode += 1
    
    def render(self):
        """Render the game"""
        # Clear screen
        self.screen.fill(Config.BACKGROUND)
        
        # Draw title
        title = self.large_font.render("Wolf & Rabbit - DQN Reinforcement Learning", 
                                      True, Config.TEXT_COLOR)
        self.screen.blit(title, (20, 10))
        
        # Draw game elements
        self.draw_grid()
        
        # Draw traps
        for trap in self.env.traps:
            self.draw_trap(trap)
        
        # Draw food
        for food in self.env.foods:
            self.draw_food(food)
        
        # Draw rabbit
        self.draw_agent(self.env.rabbit_pos, Config.RABBIT_COLOR, 0.8, "Rabbit")
        
        # Draw wolf
        self.draw_agent(self.env.wolf_pos, Config.WOLF_COLOR, 1.0, "Wolf (DQN)")
        
        # Draw info panel
        self.draw_info_panel()
        
        # Draw buttons
        self.draw_buttons()
        
        # Update display
        pygame.display.flip()
    
    def train(self, num_episodes=Config.NUM_EPISODES):
        """Main training loop"""
        print("=" * 60)
        print("Wolf & Rabbit - DQN Training Started")
        print("=" * 60)
        print(f"Device: {Config.DEVICE}")
        print(f"Grid Size: {Config.GRID_SIZE}x{Config.GRID_SIZE}")
        print(f"State Size: {Config.STATE_SIZE}, Action Size: {Config.ACTION_SIZE}")
        print(f"Episodes: {num_episodes}")
        print("\nControls:")
        print("  - ESC: Exit")
        print("  - SPACE: Pause/Resume")
        print("  - Click buttons: Pause/Save/Plot")
        print("=" * 60)
        
        while self.running and self.current_episode < num_episodes:
            self.run_episode()
            
            # Save checkpoint
            if self.current_episode % Config.SAVE_FREQUENCY == 0:
                self.agent.save_model(f"dqn_model_episode_{self.current_episode}.pth")
                self.agent.save_training_data()
        
        # Save final model and data
        self.agent.save_model("dqn_model_final.pth")
        self.agent.save_training_data()
        self.plot_training_results()
        
        print("=" * 60)
        print("Training Complete!")
        print(f"Total Episodes: {self.current_episode}")
        print(f"Total Steps: {self.total_steps}")
        print(f"Final Epsilon: {self.agent.epsilon:.4f}")
        print("=" * 60)
        
        # Keep window open
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
        
        pygame.quit()


# ==================== MAIN FUNCTION ====================
def main():
    """Main entry point"""
    try:
        # Create and run game
        game = WolfRabbitGame()
        game.train(num_episodes=Config.NUM_EPISODES)

        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        pygame.quit()


if __name__ == "__main__":
    main()