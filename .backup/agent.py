#!/usr/bin/env python3
"""
Agent Q-learning pour le jeu Snake
"""

import random
import pickle
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
from game_data import Game


class QLearning:
    """
        Q-learning agent has been implemented in traditional RL Q-learning algorithm.
        DQN has been tested but the algorithm was so slow that it was not worth it.
        DQN is adapted more for a specific situation, like a game with a lot of states and actions.
    """
    
    ACTIONS = [
        'up', 
        'down', 
        'left', 
        'right'
    ]
    
    def ft_initialize_variables(self) -> None:
        """
            Logic:
            - Initialize the variables
            - Returns the variables

            Returns:
                Nothing
        """
        self.q_table = {}
        self.learning_mode = True
        self.episodes_trained = 0
        self.total_rewards = []
        self.last_action = 'up'
        self.bonus_achieved = set()  # Pour tracker les bonus atteints
        self.total_reward = 0
        self.steps = 0
        self.timesteps_reward = 0
        self.max_length = 0
    
    def __init__(self, game: Game, learning_rate: float = 0.01, gamma: float = 0.9, 
                 epsilon: float = 0.1, epsilon_decay: float = 0.999, epsilon_min: float = 0.01) -> None:
        """
            Logic:
            - Initialize the agent
            - Returns the agent

            Returns:
                The agent
        """
        self.game = game
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.ft_initialize_variables()
    

    def ft_update_q_value(self, vision: str, action: str, reward: float, next_vision: str, done: bool) -> None:
        if not self.learning_mode:
            return
        
        current_q = self.q_table[vision][action]
        
        if done:
            target_q = reward
        else:   
            if next_vision not in self.q_table:
                self.q_table[next_vision] = self.ft_initialize_vision_q_table(next_vision)
            
            max_next_q = max(self.q_table[next_vision].values())
            target_q = reward + self.gamma * max_next_q
        
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self.q_table[vision][action] = new_q

    
    

    def ft_get_stats(self) -> dict:
        stats = {
            'total_reward': self.total_reward,
            'steps': self.steps,
            'episode': self.episodes_trained,
            'max_length': self.max_length,
            'max_length_session': self.max_length_session,
            'is_over': self.game.is_over
        }
        return stats

    def ft_check_bonus(self, session: int) -> None:
        """
            Logic:
            - Check if the bonus has been achieved
            - Returns the bonus

            Returns:
                The bonus
        """
        current_length = self.max_length
        bonus_levels = [10, 15, 20, 25, 30, 35]
        
        for bonus in bonus_levels:
            if current_length >= bonus and bonus not in self.bonus_achieved:
                print(f"🎉 BONUS ACHIEVED: Length {bonus} reached at session {session}!")
                print(f"🏆 Current maximum length: {current_length}")
                self.bonus_achieved.add(bonus)
        
        if session % 50 == 0:
            stats = self.ft_get_stats()
            print(f"\n📊 SESSION {session} - STATISTICS:")
            print(f"  Max length: {stats['max_length']} at session {stats['max_length_session']}")
            print(f"  Total reward: {stats['total_reward']:.2f}")
            print(f"  Steps: {stats['steps']}")
            print(f"  Epsilon: {self.epsilon:.4f}")
            print(f"  Learned states: {len(self.q_table)}")
            print(f"  Over: {stats['is_over']}")
            
            if self.bonus_achieved:
                print(f"  Bonuses achieved: {sorted(self.bonus_achieved)}")
            print()

    def ft_get_safe_actions(self, vision: str) -> List[str]:    
        safe_actions = []
        directions = ['up', 'down', 'left', 'right']
        
        for i, direction in enumerate(directions):
            if vision[i] not in ['W', 'S']:
                safe_actions.append(direction)
        
        return safe_actions

    def ft_initialize_vision_q_table(self, vision: str) -> None:
        q_table = {}
        q_table[vision] = {}
        for action in self.ACTIONS:
            q_table[vision][action] = 0.0
        return q_table[vision]

    def ft_choose_action(self, vision: str) -> str:
        u = random.random()
        
        if vision not in self.q_table:
            self.q_table[vision] = self.ft_initialize_vision_q_table(vision)
        
        safe_actions = self.ft_get_safe_actions(vision)
        if not safe_actions:
            safe_actions = self.ACTIONS
        
        if self.learning_mode and u < self.epsilon:
            action = random.choice(safe_actions)
        else:
            q_values = self.q_table[vision]
            self.safe_q_values = {}
            for actions in safe_actions:
                self.safe_q_values[actions] = q_values[actions]
            max_q_value = max(self.safe_q_values.values())
            best_actions = []
            for action, q_value in self.safe_q_values.items():
                if q_value == max_q_value:
                    best_actions.append(action)
            action = random.choice(best_actions)
        
        self.last_action = action
        return action


    def ft_calculate_reward(self, code: int, is_over: bool) -> float:
        if is_over:
            return -100.0
        elif code == Game.GREEN_APPLE:
            return +10.0
        elif code == Game.RED_APPLE:
            return -10.0
        else:
            return -1.0 
   
    def ft_train_agent(self, game: Game, sessions: int = 1000) -> dict:
        game.ft_reset()
        max_timesteps = 1000
        self.max_length = 0
        self.max_length_session = 0

        for session in range(sessions):
            game.ft_reset()
            self.steps = 0
            self.timesteps_reward = 0                        

            while not game.is_over and self.steps < max_timesteps:           
                vision = game.ft_get_vision_snake()
                action = self.ft_choose_action(vision)
                
                code, is_over = game.ft_move_snake(action, "off")
                reward = self.ft_calculate_reward(code, is_over)
                self.timesteps_reward += reward
                self.total_reward += reward
                next_vision = game.ft_get_vision_snake()
            
                self.ft_update_q_value(vision, action, reward, next_vision, is_over)
                
                current_length = len(game.snake)
                if current_length > self.max_length:
                    self.max_length = current_length
                    self.max_length_session = session
                
                self.steps += 1
            
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            self.episodes_trained += 1
            self.total_rewards.append(self.timesteps_reward)
            self.ft_check_bonus(session)         
    
    def set_learning_mode(self, learning: bool) -> None:
        self.learning_mode = learning
    
    def save_model(self, filepath: str) -> None:
        model_data = {
            'q_table': self.q_table,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'episodes_trained': self.episodes_trained,
            'total_rewards': self.total_rewards
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.q_table = model_data['q_table']
            self.learning_rate = model_data.get('learning_rate', self.learning_rate)
            self.gamma = model_data.get('gamma', self.gamma)
            self.epsilon = model_data.get('epsilon', self.epsilon)
            self.epsilon_decay = model_data.get('epsilon_decay', self.epsilon_decay)
            self.epsilon_min = model_data.get('epsilon_min', self.epsilon_min)
            self.episodes_trained = model_data.get('episodes_trained', 0)
            self.total_rewards = model_data.get('total_rewards', [])
            
            print(f"Model loaded from {filepath}")
            print(f"Episodes trained: {self.episodes_trained}")
            print(f"States in Q-table: {len(self.q_table)}")
            
        except FileNotFoundError:
            print(f"Model file {filepath} not found")
        except Exception as e:
            print(f"Error loading model: {e}")
  
    ################# HINTS METHODS #################
    
    game: Game
    learning_rate: float
    gamma: float
    epsilon: float
    epsilon_decay: float
    epsilon_min: float
    q_table: Dict[str, Dict[str, float]]
    learning_mode: bool
    episodes_trained: int
    total_rewards: List[float]
    last_action: str


def main():
    size = 10
    game = Game(size, size)
    agent = QLearning(game)
    sessions = 1500

    agent.load_model('./models/1500sess.txt')
    agent.set_learning_mode(False)
    agent.ft_train_agent(game, sessions)
    # agent.save_model('./models/1500sess.txt')

    


if __name__ == "__main__":
    main()