#!/usr/bin/env python3
"""
Test simple de l'agent Q-learning avec game_data.py
"""

import random
import numpy as np
from game_data import Game
from agent import QLearningAgent

def test_agent_performance():
    """Test de performance de l'agent Q-learning"""
    print("🧪 TEST DE PERFORMANCE DE L'AGENT Q-LEARNING")
    print("=" * 60)
    
    # Créer le jeu et l'agent
    size = 10
    game = Game(size, size)
    agent = QLearningAgent(game)
    
    print(f"📊 Taille du plateau: {size}x{size}")
    print(f"📊 Longueur initiale du serpent: {len(game.snake)}")
    
    # Test de quelques sessions
    max_length_achieved = 0
    sessions_to_test = 500
    
    for session in range(1, sessions_to_test + 1):
        # Reset du jeu
        game.ft_reset()
        
        # État initial
        current_state_vision = game.ft_get_vision_snake()
        current_state_encoded = agent.ft_encode_vision_simple(current_state_vision)
        agent.states = current_state_encoded
        
        # Test de quelques mouvements
        steps = 0
        max_steps = 200
        
        while not game.is_over and steps < max_steps:
            steps += 1
            
            # Choisir une action
            action_index = agent.ft_choose_action(None, agent.epsilon)
            action = agent.ACTIONS[action_index]
            
            # Exécuter l'action
            len_before = len(game.snake)
            event_code, is_over = game.ft_move_snake(action, "off")
            len_after = len(game.snake)
            
            # Calculer la récompense
            next_state_vision = game.ft_get_vision_snake()
            next_state_encoded = agent.ft_encode_vision_simple(next_state_vision)
            next_state = next_state_encoded
            
            # Calculer la récompense
            reward = agent.ft_calculate_reward(event_code, is_over, next_state, len_after)
            
            # Apprentissage Q-learning
            agent.ft_agent_learn(agent.states, action, reward, next_state, is_over)
            
            agent.states = next_state
            
            if len_after > max_length_achieved:
                max_length_achieved = len_after
                print(f"  🏆 NOUVEAU RECORD: {max_length_achieved} longueurs !")
            
            if is_over:
                break
        
        # Décroissance de l'epsilon
        agent.decay_epsilon()
        
        if session % 50 == 0:
            print(f"Session {session}: Longueur max = {max_length_achieved}, Epsilon = {agent.epsilon:.4f}")
    
    print(f"\n🎯 RÉSULTATS FINAUX:")
    print(f"📊 Longueur maximale atteinte: {max_length_achieved}")
    print(f"📊 Q-table size: {len(agent.q_table)}")
    print(f"📊 Epsilon final: {agent.epsilon:.4f}")
    
    return max_length_achieved

if __name__ == "__main__":
    test_agent_performance()
