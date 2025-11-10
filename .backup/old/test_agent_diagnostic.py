#!/usr/bin/env python3
"""
Test de diagnostic approfondi pour l'agent DQN
Permet d'identifier les problèmes d'apprentissage avec des logs détaillés
"""

import numpy as np
import pytest
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt
import sys
import os

# Ajouter le répertoire courant au path pour importer l'agent
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent import DeepQNetwork
from game_data import Game

class DiagnosticAgent(DeepQNetwork):
    """Agent avec logs détaillés pour le diagnostic"""
    
    def __init__(self, game):
        super().__init__(game)
        self.loss_history = []
        self.gradient_norms = []
        self.q_value_history = []
        self.reward_history = []
        self.epsilon_history = []
        self.learning_rates = []
        
    def ft_agent_learn(self):
        """Version avec logs détaillés de l'apprentissage"""
        if len(self.experience) < max(self.minibatch_size, self.warmup_steps):
            return None
        
        self.train_step_count += 1
        
        # Log des Q-values avant apprentissage
        if self.train_step_count % 10 == 0:
            sample_states = np.array([exp[0] for exp in list(self.experience)[-10:]])
            q_values = self.q_network.predict(sample_states, verbose=0)
            self.q_value_history.append({
                'step': self.train_step_count,
                'q_values_mean': np.mean(q_values),
                'q_values_std': np.std(q_values),
                'q_values_min': np.min(q_values),
                'q_values_max': np.max(q_values)
            })
        
        with tf.GradientTape() as tape:
            loss = self.ft_compute_loss()
        
        # Calculer les gradients
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        
        # Log des gradients
        if gradients is not None:
            grad_norms = [tf.norm(g).numpy() for g in gradients if g is not None]
            if grad_norms:
                self.gradient_norms.append({
                    'step': self.train_step_count,
                    'grad_norm_mean': np.mean(grad_norms),
                    'grad_norm_max': np.max(grad_norms),
                    'grad_norm_min': np.min(grad_norms)
                })
        
        # Appliquer les gradients
        if gradients is not None:
            self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
        
        # Soft update
        self.ft_update_target_soft_update()
        
        # Log de la loss
        loss_value = float(loss)
        self.loss_history.append({
            'step': self.train_step_count,
            'loss': loss_value
        })
        
        # Log du learning rate
        self.learning_rates.append({
            'step': self.train_step_count,
            'lr': float(self.optimizer.learning_rate)
        })
        
        if self.train_step_count % 100 == 0:
            print(f"[DIAGNOSTIC] Step {self.train_step_count}: Loss={loss_value:.4f}, "
                  f"Epsilon={self.epsilon:.4f}, Buffer={len(self.experience)}")
            if self.gradient_norms:
                last_grad = self.gradient_norms[-1]
                print(f"  Gradients: mean={last_grad['grad_norm_mean']:.4f}, "
                      f"max={last_grad['grad_norm_max']:.4f}")
        
        return loss_value

def test_agent_learning_diagnostic():
    """Test principal de diagnostic de l'apprentissage"""
    print("\n" + "="*60)
    print("DIAGNOSTIC DE L'APPRENTISSAGE DE L'AGENT DQN")
    print("="*60)
    
    # Créer le jeu et l'agent
    game = Game(10, 10)
    agent = DiagnosticAgent(game)
    
    # Configuration pour test rapide
    agent.warmup_steps = 50
    agent.epsilon = 0.5  # Exploration modérée
    agent.epsilon_decay = 0.99
    agent.epsilon_min = 0.01
    
    print(f"Configuration:")
    print(f"  - Warmup steps: {agent.warmup_steps}")
    print(f"  - Epsilon initial: {agent.epsilon}")
    print(f"  - Learning rate: {agent.alpha}")
    print(f"  - Gamma: {agent.gamma}")
    print(f"  - Batch size: {agent.minibatch_size}")
    
    # Test d'encodage de la vision
    print(f"\nTest d'encodage de la vision:")
    vision = game.ft_get_vision_snake()
    print(f"  Vision brute: {vision}")
    encoded = agent.ft_encode_vision(vision)
    print(f"  Vision encodée: shape={encoded.shape}, values={encoded}")
    
    # Test de quelques épisodes d'entraînement
    print(f"\nEntraînement de diagnostic (200 steps):")
    
    max_length_achieved = 3
    total_rewards = []
    
    for episode in range(5):  # 5 épisodes de test
        game.ft_reset()
        state = agent.ft_encode_vision(game.ft_get_vision_snake())
        agent.state_size = (state.shape[0],)
        
        if agent.q_network is None:
            agent.ft_define_initial_model()
            print(f"  Réseau créé: input_shape={agent.state_size}")
        
        episode_reward = 0
        episode_length = 0
        
        for step in range(40):  # 40 steps par épisode
            # Choisir action
            action = agent.ft_choose_action(state, agent.epsilon)
            
            # Exécuter action
            len_before = len(game.snake)
            code, is_over = game.ft_move_snake(agent.ACTIONS[action], "off")
            len_after = len(game.snake)
            
            # Calculer récompense
            reward = agent.ft_calculate_reward(code, is_over, state)
            episode_reward += reward
            
            # Obtenir nouvel état
            if not is_over:
                next_state = agent.ft_encode_vision(game.ft_get_vision_snake())
            else:
                next_state = state
            
            # Stocker expérience
            agent.ft_store_experience(state, action, reward, next_state, is_over)
            
            # Apprendre
            if agent.ft_check_update_conditions(step):
                loss = agent.ft_agent_learn()
            
            # Mettre à jour état
            state = next_state
            episode_length = len_after
            
            if len_after > max_length_achieved:
                max_length_achieved = len_after
            
            if is_over:
                break
        
        # Décroissance epsilon
        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)
        
        total_rewards.append(episode_reward)
        agent.reward_history.append({
            'episode': episode,
            'reward': episode_reward,
            'length': episode_length,
            'epsilon': agent.epsilon
        })
        
        print(f"  Épisode {episode+1}: reward={episode_reward:.2f}, "
              f"length={episode_length}, epsilon={agent.epsilon:.3f}")
    
    # Analyse des résultats
    print(f"\n" + "="*60)
    print("ANALYSE DES RÉSULTATS")
    print("="*60)
    
    print(f"Longueur maximale atteinte: {max_length_achieved}")
    print(f"Récompenses moyennes: {np.mean(total_rewards):.2f}")
    print(f"Épsilon final: {agent.epsilon:.4f}")
    print(f"Nombre d'expériences stockées: {len(agent.experience)}")
    print(f"Nombre d'étapes d'apprentissage: {agent.train_step_count}")
    
    # Analyse de la loss
    if agent.loss_history:
        losses = [h['loss'] for h in agent.loss_history]
        print(f"\nAnalyse de la loss:")
        print(f"  Loss initiale: {losses[0]:.4f}")
        print(f"  Loss finale: {losses[-1]:.4f}")
        print(f"  Loss moyenne: {np.mean(losses):.4f}")
        print(f"  Loss min: {np.min(losses):.4f}")
        print(f"  Loss max: {np.max(losses):.4f}")
        
        # Vérifier si la loss diminue
        if len(losses) > 10:
            early_loss = np.mean(losses[:5])
            late_loss = np.mean(losses[-5:])
            print(f"  Loss début (5 premiers): {early_loss:.4f}")
            print(f"  Loss fin (5 derniers): {late_loss:.4f}")
            if late_loss < early_loss:
                print(f"  ✓ La loss diminue (bon signe)")
            else:
                print(f"  ✗ La loss ne diminue pas (problème)")
    
    # Analyse des gradients
    if agent.gradient_norms:
        grad_norms = [h['grad_norm_mean'] for h in agent.gradient_norms]
        print(f"\nAnalyse des gradients:")
        print(f"  Norme moyenne: {np.mean(grad_norms):.4f}")
        print(f"  Norme max: {np.max(grad_norms):.4f}")
        print(f"  Norme min: {np.min(grad_norms):.4f}")
        
        if np.max(grad_norms) < 1e-6:
            print(f"  ✗ Gradients très petits (vanishing gradients)")
        elif np.max(grad_norms) > 100:
            print(f"  ✗ Gradients très grands (exploding gradients)")
        else:
            print(f"  ✓ Gradients dans une plage normale")
    
    # Analyse des Q-values
    if agent.q_value_history:
        q_means = [h['q_values_mean'] for h in agent.q_value_history]
        print(f"\nAnalyse des Q-values:")
        print(f"  Q-value moyenne initiale: {q_means[0]:.4f}")
        print(f"  Q-value moyenne finale: {q_means[-1]:.4f}")
        print(f"  Q-value moyenne globale: {np.mean(q_means):.4f}")
        
        if np.std(q_means) < 0.01:
            print(f"  ✗ Q-values très similaires (pas d'apprentissage)")
        else:
            print(f"  ✓ Q-values varient (apprentissage possible)")
    
    # Diagnostic final
    print(f"\n" + "="*60)
    print("DIAGNOSTIC FINAL")
    print("="*60)
    
    issues = []
    
    if max_length_achieved <= 5:
        issues.append("Longueur maximale très faible (≤5)")
    
    if agent.loss_history and len(agent.loss_history) > 10:
        losses = [h['loss'] for h in agent.loss_history]
        if np.mean(losses[-5:]) >= np.mean(losses[:5]):
            issues.append("La loss ne diminue pas")
    
    if agent.gradient_norms:
        grad_norms = [h['grad_norm_mean'] for h in agent.gradient_norms]
        if np.max(grad_norms) < 1e-6:
            issues.append("Gradients trop petits (vanishing gradients)")
        elif np.max(grad_norms) > 100:
            issues.append("Gradients trop grands (exploding gradients)")
    
    if agent.q_value_history:
        q_means = [h['q_values_mean'] for h in agent.q_value_history]
        if np.std(q_means) < 0.01:
            issues.append("Q-values trop similaires (pas d'apprentissage)")
    
    if not issues:
        print("✓ Aucun problème majeur détecté")
    else:
        print("✗ Problèmes détectés:")
        for issue in issues:
            print(f"  - {issue}")
    
    return max_length_achieved, issues

def test_vision_encoding():
    """Test spécifique de l'encodage de la vision"""
    print("\n" + "="*40)
    print("TEST D'ENCODAGE DE LA VISION")
    print("="*40)
    
    game = Game(10, 10)
    agent = DeepQNetwork(game)
    
    # Test avec différents états de vision
    test_cases = [
        {"up": [0, 0, 0], "down": [1, 0, 0], "left": [0, 0, 0], "right": [0, 0, 0]},
        {"up": [4, 0, 0], "down": [0, 0, 0], "left": [0, 0, 0], "right": [0, 0, 0]},
        {"up": [5, 0, 0], "down": [0, 0, 0], "left": [0, 0, 0], "right": [0, 0, 0]},
        {"up": [], "down": [], "left": [], "right": []},
    ]
    
    for i, vision in enumerate(test_cases):
        encoded = agent.ft_encode_vision(vision)
        print(f"Test {i+1}: {vision}")
        print(f"  Encodé: shape={encoded.shape}, values={encoded}")
        print()

if __name__ == "__main__":
    # Exécuter les tests de diagnostic
    test_vision_encoding()
    max_length, issues = test_agent_learning_diagnostic()
    
    print(f"\nRésultat final: Longueur max = {max_length}")
    if issues:
        print(f"Problèmes: {len(issues)}")
    else:
        print("Aucun problème détecté")
