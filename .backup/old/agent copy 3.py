#!/usr/bin/env python3
"""
Module Agent - Q-learning agent for the Snake game
Implements Q-learning algorithm with exploration/exploitation
"""

import random
import json
import pickle
from typing import Dict, Tuple, Optional
import numpy as np


class QLearningAgent:
    """Q-learning agent for the Snake game"""
    
    # Actions
    ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.9, 
                 epsilon: float = 0.1, epsilon_decay: float = 0.999, 
                 epsilon_min: float = 0.01):
        """
        Initialize the Q-learning agent
        
        Args:
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Exploration rate
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon value
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
    
    def get_state_key(self, state: str) -> str:
        """
        Convert state to a key for the Q-table
        State is the snake's vision in 4 directions
        """
        return state
    
    def get_q_value(self, state: str, action: str) -> float:
        """Get Q-value for a state-action pair"""
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0.0 for action in self.ACTIONS}
        return self.q_table[state_key].get(action, 0.0)
    
    def set_q_value(self, state: str, action: str, value: float):
        """Set Q-value for a state-action pair"""
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0.0 for action in self.ACTIONS}
        self.q_table[state_key][action] = value
    
    def choose_action(self, state: str) -> str:
        """
        Choose an action using epsilon-greedy policy with wall avoidance
        
        Args:
            state: Current state (snake vision)
            
        Returns:
            Chosen action
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
        
        Args:
            state: Current state (snake vision in 4 directions)
            
        Returns:
            List of safe actions
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
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is finished
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
        """Decay epsilon for exploration"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def set_learning_mode(self, learning: bool):
        """Set learning mode on/off"""
        self.learning_mode = learning
    
    def save_model(self, filepath: str):
        """Save the trained model to a file"""
        model_data = {
            'q_table': self.q_table,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
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
        """Load a trained model from a file"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.q_table = model_data['q_table']
            self.learning_rate = model_data.get('learning_rate', self.learning_rate)
            self.discount_factor = model_data.get('discount_factor', self.discount_factor)
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
    
    def get_stats(self) -> dict:
        """Get agent statistics"""
        return {
            'episodes_trained': self.episodes_trained,
            'epsilon': self.epsilon,
            'states_learned': len(self.q_table),
            'learning_mode': self.learning_mode,
            'avg_reward': np.mean(self.total_rewards) if self.total_rewards else 0
        }
    
    def train_episode(self, board, max_steps: int = 1000) -> dict:
        """
        Train the agent for one episode
        
        Args:
            board: Game board
            max_steps: Maximum steps per episode
            
        Returns:
            Episode statistics
        """
        board.reset()
        total_reward = 0
        steps = 0
        episode_reward = 0
        
        while not board.is_game_over() and steps < max_steps:
            # Get current state
            state = board.get_state()
            
            # Choose action
            action = self.choose_action(state)
            
            # Take action
            game_continues, reward = board.move_snake(action)
            episode_reward += reward
            total_reward += reward
            
            # Get next state
            next_state = board.get_state() if not board.is_game_over() else ""
            
            # Update Q-value
            self.update_q_value(state, action, reward, next_state, board.is_game_over())
            
            steps += 1
        
        # Decay epsilon
        self.decay_epsilon()
        
        # Update statistics
        self.episodes_trained += 1
        self.total_rewards.append(total_reward)
        self.episode_rewards.append(episode_reward)
        
        stats = board.get_stats()
        stats.update({
            'total_reward': total_reward,
            'steps': steps,
            'episode': self.episodes_trained
        })
        
        return stats
    
    def evaluate_episode(self, board, max_steps: int = 1000) -> dict:
        """
        Evaluate the agent without learning
        
        Args:
            board: Game board
            max_steps: Maximum steps per episode
            
        Returns:
            Episode statistics
        """
        # Save current learning mode
        old_learning_mode = self.learning_mode
        
        # Disable learning
        self.set_learning_mode(False)
        
        # Run episode
        stats = self.train_episode(board, max_steps)
        
        # Restore learning mode
        self.set_learning_mode(old_learning_mode)
        
        return stats


def test_bonus_lengths():
    """Test des longueurs bonus avec l'agent Q-learning"""
    from AI.board import Board
    
    print("🧪 TEST DES LONGUEURS BONUS")
    print("=" * 50)
    
    agent = QLearningAgent(
        learning_rate=0.2,
        discount_factor=0.95,
        epsilon=0.3,
        epsilon_decay=0.999,
        epsilon_min=0.01
    )
    board = Board()

    # Longueurs à tester pour le bonus
    test_lengths = [15, 20, 25, 30, 35]
    longueur_atteinte = set()
    max_length_global = 0
    
    # Statistiques de progression
    progress_stats = {
        'sessions_10': 0,
        'sessions_15': 0,
        'sessions_20': 0,
        'sessions_25': 0,
        'sessions_30': 0,
        'sessions_35': 0
    }

    # Train for multiple episodes
    num_episodes = 500
    print(f"🚀 Démarrage de l'entraînement sur {num_episodes} épisodes...")
    
    for episode in range(1, num_episodes + 1):
        stats = agent.train_episode(board, max_steps=1000)
        length = stats.get('max_length', 0)
        
        # Mise à jour de la longueur maximale
        if length > max_length_global:
            max_length_global = length
            print(f"🏆 NOUVEAU RECORD: {max_length_global} longueurs à l'épisode {episode} !")

        # Vérifie les paliers de longueur bonus
        for palier in test_lengths:
            if length >= palier and palier not in longueur_atteinte:
                print(f"🎉 BONUS: Longueur {palier} atteinte à l'épisode {episode} !")
                longueur_atteinte.add(palier)
                progress_stats[f'sessions_{palier}'] = episode

        # Affichage des statistiques tous les 50 épisodes
        if episode % 50 == 0:
            print(f"\n📊 ÉPISODE {episode} - STATISTIQUES:")
            print(f"  Longueur max globale: {max_length_global}")
            print(f"  Longueur actuelle: {length}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  États appris: {len(agent.q_table)}")
            print(f"  Récompense totale: {stats.get('total_reward', 0):.2f}")
            print(f"  Étapes: {stats.get('steps', 0)}")
            
            # Afficher les paliers atteints
            if longueur_atteinte:
                print(f"  Paliers atteints: {sorted(longueur_atteinte)}")
            print()

    # Résumé final
    print("\n🎯 RÉSUMÉ FINAL DU TEST BONUS:")
    print("=" * 50)
    print(f"📊 Longueur maximale atteinte: {max_length_global}")
    print(f"📊 États appris dans Q-table: {len(agent.q_table)}")
    print(f"📊 Epsilon final: {agent.epsilon:.4f}")
    
    print(f"\n🏆 PALIERS BONUS ATTEINTS:")
    for palier in test_lengths:
        if palier in longueur_atteinte:
            episode = progress_stats[f'sessions_{palier}']
            print(f"  ✅ Longueur {palier}: Épisode {episode}")
        else:
            print(f"  ❌ Longueur {palier}: Non atteint")
    
    # Test de performance finale (5 épisodes sans apprentissage)
    print(f"\n🧪 TEST DE PERFORMANCE FINALE (5 épisodes sans apprentissage):")
    agent.set_learning_mode(False)
    final_lengths = []
    for i in range(5):
        stats = agent.evaluate_episode(board, max_steps=1000)
        length = stats.get('max_length', 0)
        final_lengths.append(length)
        print(f"  Épisode {i+1}: {length} longueurs")
    
    print(f"📊 Moyenne finale: {np.mean(final_lengths):.2f}")
    print(f"📊 Maximum final: {max(final_lengths)}")
    
    return max_length_global, longueur_atteinte

def main():
    """Fonction principale avec test des bonus"""
    try:
        max_length, bonus_achieved = test_bonus_lengths()
        
        print(f"\n🎉 RÉSULTAT FINAL:")
        print(f"Longueur maximale: {max_length}")
        print(f"Bonus atteints: {len(bonus_achieved)}/5")
        
        if len(bonus_achieved) >= 3:
            print("✅ SUCCÈS: Plus de la moitié des bonus atteints !")
        else:
            print("⚠️  PARTIEL: Certains bonus non atteints")
            
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

