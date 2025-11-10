#!/usr/bin/env python3
"""
Test complet de AI/board.py pour vérifier son fonctionnement
"""

import random
import numpy as np
from AI.board import Board

def test_board_basic():
    """Test de base de AI/board.py"""
    print("🧪 TEST DE BASE DE AI/BOARD.PY")
    print("=" * 50)
    
    board = Board()
    print(f"📊 Taille du plateau: {board.width}x{board.height}")
    print(f"📊 Longueur initiale du serpent: {len(board.snake)}")
    print(f"📊 Pommes vertes: {np.sum(board.board == board.GREEN_APPLE)}")
    print(f"📊 Pommes rouges: {np.sum(board.board == board.RED_APPLE)}")
    
    # Test de la vision
    vision = board.get_snake_vision()
    print(f"📊 Vision initiale: '{vision}'")
    
    return board

def test_board_vision():
    """Test de la vision du serpent"""
    print("\n🔍 TEST DE LA VISION DU SERPENT")
    print("=" * 50)
    
    board = Board()
    
    # Test de plusieurs positions
    for i in range(5):
        board.reset()
        vision = board.get_snake_vision()
        print(f"Position {i+1}: Vision = '{vision}' (longueur: {len(vision)})")
        
        # Vérifier que la vision a 4 caractères
        assert len(vision) == 4, f"Vision doit avoir 4 caractères, a {len(vision)}"
        
        # Vérifier que chaque caractère est valide
        valid_chars = ['0', 'W', 'H', 'S', 'G', 'R']  # '0' pour EMPTY dans AI/board.py
        for j, char in enumerate(vision):
            assert char in valid_chars, f"Caractère invalide '{char}' en position {j}"

def test_board_movement():
    """Test des mouvements du serpent"""
    print("\n🚀 TEST DES MOUVEMENTS DU SERPENT")
    print("=" * 50)
    
    board = Board()
    
    # Test de plusieurs mouvements
    for i in range(10):
        vision_before = board.get_snake_vision()
        len_before = len(board.snake)
        
        # Choisir une action aléatoire
        action = random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])
        
        # Exécuter l'action
        game_continues, reward = board.move_snake(action)
        
        vision_after = board.get_snake_vision()
        len_after = len(board.snake)
        
        print(f"Action {i+1}: {action}")
        print(f"  Vision: '{vision_before}' -> '{vision_after}'")
        print(f"  Longueur: {len_before} -> {len_after}")
        print(f"  Récompense: {reward}")
        print(f"  Jeu continue: {game_continues}")
        
        if not game_continues:
            print(f"  💀 GAME OVER après {i+1} actions")
            break
        
        print()

def test_board_apple_eating():
    """Test de la consommation de pommes"""
    print("\n🍎 TEST DE LA CONSOMMATION DE POMMES")
    print("=" * 50)
    
    board = Board()
    
    # Chercher une pomme verte
    green_apples = np.argwhere(board.board == board.GREEN_APPLE)
    if len(green_apples) > 0:
        print(f"Pommes vertes trouvées: {len(green_apples)}")
        
        # Essayer de se diriger vers une pomme verte
        for i in range(20):
            vision = board.get_snake_vision()
            len_before = len(board.snake)
            
            # Choisir une action qui pourrait nous rapprocher d'une pomme verte
            if 'G' in vision:
                # Si on voit une pomme verte, aller dans cette direction
                if vision[0] == 'G':  # UP
                    action = 'UP'
                elif vision[1] == 'G':  # DOWN
                    action = 'DOWN'
                elif vision[2] == 'G':  # LEFT
                    action = 'LEFT'
                elif vision[3] == 'G':  # RIGHT
                    action = 'RIGHT'
                else:
                    action = random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])
            else:
                action = random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])
            
            game_continues, reward = board.move_snake(action)
            len_after = len(board.snake)
            
            print(f"Action {i+1}: {action} -> Récompense: {reward}, Longueur: {len_before} -> {len_after}")
            
            if reward > 0:
                print(f"  🎉 POMME VERTE MANGÉE ! Longueur: {len_after}")
            
            if not game_continues:
                print(f"  💀 GAME OVER")
                break
    else:
        print("Aucune pomme verte trouvée")

def test_board_red_apple_death():
    """Test de la mort par pomme rouge"""
    print("\n🍎 TEST DE LA MORT PAR POMME ROUGE")
    print("=" * 50)
    
    board = Board()
    
    # Chercher une pomme rouge
    red_apples = np.argwhere(board.board == board.RED_APPLE)
    if len(red_apples) > 0:
        print(f"Pommes rouges trouvées: {len(red_apples)}")
        
        # Essayer de se diriger vers une pomme rouge
        for i in range(20):
            vision = board.get_snake_vision()
            len_before = len(board.snake)
            
            # Choisir une action qui pourrait nous rapprocher d'une pomme rouge
            if 'R' in vision:
                # Si on voit une pomme rouge, aller dans cette direction
                if vision[0] == 'R':  # UP
                    action = 'UP'
                elif vision[1] == 'R':  # DOWN
                    action = 'DOWN'
                elif vision[2] == 'R':  # LEFT
                    action = 'LEFT'
                elif vision[3] == 'R':  # RIGHT
                    action = 'RIGHT'
                else:
                    action = random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])
            else:
                action = random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])
            
            game_continues, reward = board.move_snake(action)
            len_after = len(board.snake)
            
            print(f"Action {i+1}: {action} -> Récompense: {reward}, Longueur: {len_before} -> {len_after}")
            
            if reward < 0:
                print(f"  ⚠️ POMME ROUGE MANGÉE ! Longueur: {len_after}")
                if len_after == 1:
                    print(f"  💀 SERPENT RÉDUIT À SA TÊTE SEULE !")
            
            if not game_continues:
                print(f"  💀 GAME OVER")
                break
    else:
        print("Aucune pomme rouge trouvée")

def test_board_wall_collision():
    """Test de collision avec les murs"""
    print("\n🧱 TEST DE COLLISION AVEC LES MURS")
    print("=" * 50)
    
    board = Board()
    
    # Essayer de se diriger vers un mur
    for i in range(20):
        vision = board.get_snake_vision()
        len_before = len(board.snake)
        
        # Choisir une action qui pourrait nous rapprocher d'un mur
        if 'W' in vision:
            # Si on voit un mur, aller dans cette direction
            if vision[0] == 'W':  # UP
                action = 'UP'
            elif vision[1] == 'W':  # DOWN
                action = 'DOWN'
            elif vision[2] == 'W':  # LEFT
                action = 'LEFT'
            elif vision[3] == 'W':  # RIGHT
                action = 'RIGHT'
            else:
                action = random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])
        else:
            action = random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])
        
        game_continues, reward = board.move_snake(action)
        len_after = len(board.snake)
        
        print(f"Action {i+1}: {action} -> Récompense: {reward}, Longueur: {len_before} -> {len_after}")
        
        if not game_continues:
            print(f"  💀 GAME OVER - Collision avec mur")
            break

def test_board_self_collision():
    """Test de collision avec soi-même"""
    print("\n🐍 TEST DE COLLISION AVEC SOI-MÊME")
    print("=" * 50)
    
    board = Board()
    
    # Essayer de se diriger vers son propre corps
    for i in range(20):
        vision = board.get_snake_vision()
        len_before = len(board.snake)
        
        # Choisir une action qui pourrait nous rapprocher de notre corps
        if 'S' in vision:
            # Si on voit notre corps, aller dans cette direction
            if vision[0] == 'S':  # UP
                action = 'UP'
            elif vision[1] == 'S':  # DOWN
                action = 'DOWN'
            elif vision[2] == 'S':  # LEFT
                action = 'LEFT'
            elif vision[3] == 'S':  # RIGHT
                action = 'RIGHT'
            else:
                action = random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])
        else:
            action = random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])
        
        game_continues, reward = board.move_snake(action)
        len_after = len(board.snake)
        
        print(f"Action {i+1}: {action} -> Récompense: {reward}, Longueur: {len_before} -> {len_after}")
        
        if not game_continues:
            print(f"  💀 GAME OVER - Collision avec soi-même")
            break

def test_board_detailed_vision():
    """Test de la vision détaillée"""
    print("\n🔍 TEST DE LA VISION DÉTAILLÉE")
    print("=" * 50)
    
    board = Board()
    
    # Test de la vision détaillée
    vision_detailed = board.get_snake_vision_detailed()
    print(f"Vision détaillée: {vision_detailed}")
    
    # Vérifier que chaque direction a une liste
    for direction in ['up', 'down', 'left', 'right']:
        assert direction in vision_detailed, f"Direction '{direction}' manquante"
        assert isinstance(vision_detailed[direction], list), f"Direction '{direction}' n'est pas une liste"
        print(f"  {direction}: {vision_detailed[direction]}")

def main():
    """Lancer tous les tests"""
    print("🧪 TESTS COMPLETS DE AI/BOARD.PY")
    print("=" * 60)
    
    try:
        # Test de base
        board = test_board_basic()
        
        # Test de la vision
        test_board_vision()
        
        # Test des mouvements
        test_board_movement()
        
        # Test de consommation de pommes
        test_board_apple_eating()
        
        # Test de mort par pomme rouge
        test_board_red_apple_death()
        
        # Test de collision avec les murs
        test_board_wall_collision()
        
        # Test de collision avec soi-même
        test_board_self_collision()
        
        # Test de la vision détaillée
        test_board_detailed_vision()
        
        print("\n✅ TOUS LES TESTS SONT PASSÉS !")
        
    except Exception as e:
        print(f"\n❌ ERREUR DANS LES TESTS: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
