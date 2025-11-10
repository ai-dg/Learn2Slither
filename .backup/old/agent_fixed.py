#!/usr/bin/env python3
"""
Agent Q-Learning adapté pour game_data.py - Copie exacte de AI/agent.py
"""

import random
import numpy as np
from collections import deque
from game_data import Game

class QLearningAgent:
    """Agent Q-Learning adapté pour game_data.py - Copie exacte de AI/agent.py"""
    
    # Actions (majuscules comme AI/agent.py)
    ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    
    # Mapping des actions vers game_data.py
    ACTION_MAPPING = {
        'UP': 'up',
        'DOWN': 'down', 
        'LEFT': 'left',
        'RIGHT': 'right'
    }

    def __init__(self, game: Game, learning_rate: float = 0.1, discount_factor: float = 0.9, 
                 epsilon: float = 0.1, epsilon_decay: float = 0.999, 
                 epsilon_min: float = 0.01):
        """
        Initialize the Q-learning agent (copie exacte de AI/agent.py)
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: state -> action -> Q-value
        self.q_table = {}
        
        # Learning mode
        self.learning_mode = True
        
        # Statistics
        self.episodes_trained = 0
        self.total_rewards = []
        self.episode_rewards = []
        
        # Last action taken (for terminal display)
        self.last_action = 'UP'
        
        # Game reference
        self.game = game

    def get_state_key(self, state: str) -> str:
        """Convert state to a key for the Q-table (copie exacte de AI/agent.py)"""
        return state

    def get_q_value(self, state: str, action: str) -> float:
        """Get Q-value for a state-action pair (copie exacte de AI/agent.py)"""
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0.0 for action in self.ACTIONS}
        return self.q_table[state_key].get(action, 0.0)

    def set_q_value(self, state: str, action: str, value: float):
        """Set Q-value for a state-action pair (copie exacte de AI/agent.py)"""
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0.0 for action in self.ACTIONS}
        self.q_table[state_key][action] = value

    def ft_encode_vision_simple(self, vision):
        """Encoder la vision de game_data.py en état discret (comme AI/board.py)"""
        state_chars = []
        for a in ['up', 'down', 'left', 'right']:  # Ordre des directions
            if a in vision and vision[a]:
                # Prendre le premier élément de chaque direction (le plus proche)
                first_element = vision[a][0] if vision[a] else 0
                # Convertir en caractère pour Q-table
                if first_element == 0:
                    state_chars.append('E')  # EMPTY
                elif first_element == 1:
                    state_chars.append('W')  # WALL
                elif first_element == 2:
                    state_chars.append('H')  # HEAD
                elif first_element == 3:
                    state_chars.append('S')  # BODY (Snake)
                elif first_element == 4:
                    state_chars.append('G')  # GREEN_APPLE
                elif first_element == 5:
                    state_chars.append('R')  # RED_APPLE
                else:
                    state_chars.append('E')  # EMPTY par défaut
            else:
                state_chars.append('E')  # EMPTY si pas de vision
        return ''.join(state_chars)

    def choose_action(self, state: str) -> str:
        """
        Choose an action using epsilon-greedy policy with wall avoidance
        (copie exacte de AI/agent.py)
        """
        state_key = self.get_state_key(state)
        
        # Initialize Q-values for new state
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0.0 for action in self.ACTIONS}
        
        # Get safe actions (not leading to walls)
        safe_actions = self.get_safe_actions(state)
        
        # If no safe actions, choose randomly (shouldn't happen in normal gameplay)
        if not safe_actions:
            safe_actions = self.ACTIONS
        
        # Epsilon-greedy action selection
        if self.learning_mode and random.random() < self.epsilon:
            # Explore: choose random action from safe actions
            action = random.choice(safe_actions)
        else:
            # Exploit: choose best action from safe actions
            q_values = self.q_table[state_key]
            safe_q_values = {action: q_values[action] for action in safe_actions}
            max_q_value = max(safe_q_values.values())
            
            # If multiple actions have the same max Q-value, choose randomly among them
            best_actions = [action for action, q_value in safe_q_values.items() 
                          if q_value == max_q_value]
            action = random.choice(best_actions)
        
        # Store last action for terminal display
        self.last_action = action
        return action

    def get_safe_actions(self, state: str) -> list:
        """
        Get actions that don't lead to walls or snake body
        (copie exacte de AI/agent.py)
        """
        if len(state) != 4:
            return self.ACTIONS
        
        safe_actions = []
        # state format: "WHSG" where each char represents what's in that direction
        # Directions: UP, DOWN, LEFT, RIGHT
        directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        
        for i, direction in enumerate(directions):
            if i < len(state) and state[i] not in ['W', 'S']:  # Not a wall or snake body
                safe_actions.append(direction)
        
        return safe_actions if safe_actions else self.ACTIONS

    def update_q_value(self, state: str, action: str, reward: float, 
                      next_state: str, done: bool):
        """
        Update Q-value using Q-learning algorithm
        (copie exacte de AI/agent.py)
        """
        if not self.learning_mode:
            return
        
        current_q = self.get_q_value(state, action)
        
        if done:
            # Terminal state
            target_q = reward
        else:
            # Non-terminal state
            next_state_key = self.get_state_key(next_state)
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = {action: 0.0 for action in self.ACTIONS}
            
            max_next_q = max(self.q_table[next_state_key].values())
            target_q = reward + self.discount_factor * max_next_q
        
        # Update Q-value
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self.set_q_value(state, action, new_q)

    def decay_epsilon(self):
        """Decay epsilon for exploration (copie exacte de AI/agent.py)"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train_episode(self, max_steps: int = 1000) -> dict:
        """
        Train the agent for one episode (copie exacte de AI/agent.py)
        """
        # Reset game
        self.game.ft_reset()
        
        # Get initial state
        vision = self.game.ft_get_vision_snake()
        state = self.ft_encode_vision_simple(vision)
        
        total_reward = 0
        steps = 0
        max_length = 0
        
        while not self.game.is_over and steps < max_steps:
            steps += 1
            
            # Choose action
            action = self.choose_action(state)
            
            # Convert action to game_data.py format
            game_action = self.ACTION_MAPPING[action]
            
            # Execute action
            len_before = len(self.game.snake)
            event_code, is_over = self.game.ft_move_snake(game_action, "off")
            len_after = len(self.game.snake)
            
            # Calculate reward (comme AI/board.py)
            if is_over:
                reward = -100  # Death penalty
            elif event_code == 4:  # GREEN_APPLE
                reward = 10  # Green apple reward
            elif event_code == 5:  # RED_APPLE
                reward = -10  # Red apple penalty
            else:
                reward = 0  # Normal move
            
            # Get next state
            next_vision = self.game.ft_get_vision_snake()
            next_state = self.ft_encode_vision_simple(next_vision)
            
            # Update Q-value
            self.update_q_value(state, action, reward, next_state, is_over)
            
            # Update state
            state = next_state
            total_reward += reward
            
            # Track max length
            if len_after > max_length:
                max_length = len_after
            
            if is_over:
                break
        
        # Decay epsilon
        self.decay_epsilon()
        
        # Update statistics
        self.episodes_trained += 1
        self.total_rewards.append(total_reward)
        self.episode_rewards.append(total_reward)
        
        return {
            'total_reward': total_reward,
            'steps': steps,
            'max_length': max_length,
            'final_length': len(self.game.snake)
        }

def main():
    """Test the agent with game_data.py"""
    size = 10
    game = Game(size, size)
    agent = QLearningAgent(game)
    
    print("🎯 ENTRAÎNEMENT AGENT Q-LEARNING AVEC GAME_DATA.PY")
    print("=" * 60)
    
    # Train for multiple episodes
    num_episodes = 1000
    max_length_achieved = 0
    
    for episode in range(1, num_episodes + 1):
        stats = agent.train_episode()
        
        if stats['max_length'] > max_length_achieved:
            max_length_achieved = stats['max_length']
            print(f"🏆 NOUVEAU RECORD: {max_length_achieved} longueurs !")
        
        if episode % 100 == 0:
            print(f"Episode {episode}: Max length = {max_length_achieved}, Epsilon = {agent.epsilon:.4f}")
            print(f"Q-table size: {len(agent.q_table)}")
    
    print(f"\n🎯 RÉSULTATS FINAUX:")
    print(f"📊 Longueur maximale atteinte: {max_length_achieved}")
    print(f"📊 Q-table size: {len(agent.q_table)}")
    print(f"📊 Epsilon final: {agent.epsilon:.4f}")

if __name__ == "__main__":
    main()