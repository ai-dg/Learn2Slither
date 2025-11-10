#!/usr/bin/env python3
"""
Test simple de l'agent avec game_data.py
"""

import random
import numpy as np
from game_data import Game
from agent import QLearningAgent

def test_agent_with_game_data():
    """Test simple de l'agent avec game_data.py"""
    print("🧪 TEST DE L'AGENT AVEC GAME_DATA.PY")
    print("=" * 50)
    
    # Créer le jeu et l'agent
    size = 10
    game = Game(size, size)
    agent = QLearningAgent(game)
    
    print(f"📊 Taille du plateau: {size}x{size}")
    print(f"📊 Longueur initiale du serpent: {len(game.snake)}")
    print(f"📊 Pommes vertes: {np.sum(game.map == game.GREEN_APPLE)}")
    print(f"📊 Pommes rouges: {np.sum(game.map == game.RED_APPLE)}")
    
    # Test de quelques sessions
    max_length_achieved = 0
    sessions_to_test = 10
    
    for session in range(1, sessions_to_test + 1):
        print(f"\n--- SESSION {session} ---")
        
        # Reset du jeu
        game.ft_reset()
        print(f"Longueur après reset: {len(game.snake)}")
        
        # Test de quelques mouvements
        steps = 0
        max_steps = 50
        
        while not game.is_over and steps < max_steps:
            steps += 1
            
            # Obtenir la vision
            vision = game.ft_get_vision_snake()
            print(f"  Vision: {vision}")
            
            # Choisir une action
            action_index = agent.ft_choose_action(None, agent.epsilon)
            action = agent.ACTIONS[action_index]
            print(f"  Action choisie: {action}")
            
            # Exécuter l'action
            len_before = len(game.snake)
            event_code, is_over = game.ft_move_snake(action, "off")
            len_after = len(game.snake)
            
            print(f"  Longueur: {len_before} -> {len_after}")
            print(f"  Événement: {game.CHARS[event_code] if event_code < len(game.CHARS) else 'UNKNOWN'}")
            print(f"  Game over: {is_over}")
            
            if len_after > max_length_achieved:
                max_length_achieved = len_after
                print(f"  🏆 NOUVEAU RECORD: {max_length_achieved} longueurs !")
            
            if is_over:
                print(f"  💀 GAME OVER après {steps} étapes")
                break
        
        print(f"Résultat session {session}: {len(game.snake)} longueurs, {steps} étapes")
    
    print(f"\n🎯 RÉSULTATS FINAUX:")
    print(f"📊 Longueur maximale atteinte: {max_length_achieved}")
    print(f"📊 Q-table size: {len(agent.q_table)}")
    
    return max_length_achieved

if __name__ == "__main__":
    test_agent_with_game_data()
