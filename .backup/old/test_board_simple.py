#!/usr/bin/env python3
"""
Test simple de AI/board.py pour comprendre le comportement
"""

import random
import numpy as np
from AI.board import Board

def test_board_simple():
    """Test simple de AI/board.py"""
    print("🧪 TEST SIMPLE DE AI/BOARD.PY")
    print("=" * 50)
    
    board = Board()
    
    print(f"📊 Plateau initial:")
    print(f"  Taille: {board.width}x{board.height}")
    print(f"  Serpent: {board.snake}")
    print(f"  Game over: {board.game_over}")
    
    # Afficher le plateau
    print(f"\n📊 Plateau visuel:")
    for y in range(board.height):
        line = ""
        for x in range(board.width):
            cell = board.board[y, x]
            if cell == board.EMPTY:
                line += "."
            elif cell == board.WALL:
                line += "#"
            elif cell == board.SNAKE_HEAD:
                line += "H"
            elif cell == board.SNAKE_BODY:
                line += "S"
            elif cell == board.GREEN_APPLE:
                line += "G"
            elif cell == board.RED_APPLE:
                line += "R"
        print(f"  {line}")
    
    # Test de la vision
    vision = board.get_snake_vision()
    print(f"\n📊 Vision du serpent: '{vision}'")
    
    # Test de chaque direction
    directions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    for direction in directions:
        # Reset du plateau
        board.reset()
        
        print(f"\n📊 Test direction {direction}:")
        print(f"  Vision avant: '{board.get_snake_vision()}'")
        print(f"  Serpent avant: {board.snake}")
        
        # Essayer de bouger
        game_continues, reward = board.move_snake(direction)
        
        print(f"  Vision après: '{board.get_snake_vision()}'")
        print(f"  Serpent après: {board.snake}")
        print(f"  Récompense: {reward}")
        print(f"  Jeu continue: {game_continues}")
        print(f"  Game over: {board.game_over}")

def test_board_with_agent():
    """Test avec un agent simple qui évite les collisions"""
    print("\n🤖 TEST AVEC AGENT SIMPLE")
    print("=" * 50)
    
    board = Board()
    
    # Agent simple qui évite les collisions
    for step in range(20):
        vision = board.get_snake_vision()
        len_before = len(board.snake)
        
        # Choisir une action sûre
        safe_actions = []
        for i, direction in enumerate(['UP', 'DOWN', 'LEFT', 'RIGHT']):
            if i < len(vision) and vision[i] not in ['W', 'S']:  # Pas de mur ou corps
                safe_actions.append(direction)
        
        if safe_actions:
            action = random.choice(safe_actions)
        else:
            action = random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])
        
        print(f"Étape {step+1}: Action = {action}, Vision = '{vision}'")
        
        game_continues, reward = board.move_snake(action)
        len_after = len(board.snake)
        
        print(f"  Récompense: {reward}, Longueur: {len_before} -> {len_after}")
        print(f"  Jeu continue: {game_continues}")
        
        if not game_continues:
            print(f"  💀 GAME OVER")
            break
        
        if reward > 0:
            print(f"  🎉 POMME MANGÉE !")
        
        print()

if __name__ == "__main__":
    test_board_simple()
    test_board_with_agent()
