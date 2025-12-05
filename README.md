# Learn2Slither 🐍

**A snake that learns how to behave in an environment through trial and error using Q-learning reinforcement learning**

> Train an AI agent to play Snake using Q-learning algorithm with limited vision and proper reward system.

---

## ▌Project Overview

This project implements a complete Snake game with an AI agent that learns to play using **Q-learning reinforcement learning**.\
The snake learns through trial and error, gradually improving its performance over multiple training sessions.\
It serves as an introduction to **reinforcement learning fundamentals**, written from scratch without using high-level ML libraries.

📘 Educational AI project: **you'll build and train your own Q-learning model step-by-step**.

---

## ▌Features

✔️ **Q-learning Algorithm**: Implements Q-table based reinforcement learning\
✔️ **Snake Vision**: Agent only sees 4 directions from its head (W, H, S, G, R, 0)\
✔️ **Proper Rewards**: Positive for green apples, negative for red apples and collisions\
✔️ **Model Save/Load**: Export and import trained models using pickle\
✔️ **Graphical Interface**: Pygame-based visualization with real-time display\
✔️ **Terminal Display**: Exact format as shown in subject illustrations (W, 0, G, R, H, S)\
✔️ **Step-by-step Mode**: Manual control for debugging and analysis\
✔️ **Command Line Interface**: Full CLI with all required parameters\
✔️ **Modular Architecture**: Separate modules for board, agent, and GUI

---

## ▌Fonctionnalités Bonus

- ■ **Interface Graphique Améliorée** : Système de lobby, panneau de configuration, suivi des statistiques
- ■ **Tailles de Plateau Variables** : Support pour différentes dimensions (10×10, 15×15, 20×20, jusqu'à 40×40)
- ■ **Haute Performance** : Longueur maximale atteinte de **71 cellules** après 4500 sessions
- ■ **Paramètres Avancés** : Contrôle de la vitesse (FPS), mode pas-à-pas, tracé des courbes d'apprentissage
- ■ **Statistiques Détaillées** : Affichage périodique des performances avec bonus atteints

> ⚠️ Ces fonctionnalités ne sont évaluées que si le programme de base fonctionne parfaitement.

---

## ▌How it works

### ■ Method Used

The model is trained using **Q-learning** with **epsilon-greedy exploration**. The objective is to maximize the **cumulative reward** by learning optimal actions for each state.

### ■ State Representation

The agent receives a 4-character string representing what it sees in each direction:
- `W` = Wall
- `H` = Snake Head  
- `S` = Snake Body
- `G` = Green Apple
- `R` = Red Apple
- `0` = Empty Space

### ■ Q-Learning Parameters

```text
Learning Rate: 0.1
Discount Factor: 0.9 (basic) / 0.95 (advanced)
Epsilon: 0.1 (exploration rate)
Epsilon Decay: 0.995
Epsilon Min: 0.01
```

### ■ Reward System

```text
Green Apple: +10
Red Apple: -10
Move without eating: -1
Game Over: -100
```

---

## ▌Getting Started

### ■ Requirements

- Python 3.x
- `pygame` (graphical interface)
- `numpy` (numerical operations)
- `pickle` (model serialization)
- `tabulate` (table formatting)
- `matplotlib` (plotting, optionnel)

### ■ Installation

1. Clone the repository

```bash
git clone https://github.com/ai-dg/Learn2Slither.git
cd Learn2Slither
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▌Instructions d'Utilisation

### ■ Syntaxe de Base

```bash
python3 snake.py [OPTIONS]
```

ou directement :

```bash
./snake [OPTIONS]
```

### ■ Options Disponibles

#### Options Principales

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `-sessions N` | entier | 10 | Nombre de sessions d'entraînement |
| `-visual [on\|off]` | choix | off | Active/désactive l'affichage graphique (Pygame) |
| `-terminal [on\|off]` | choix | on | Active/désactive l'affichage terminal du jeu |
| `-save PATH` | chaîne | - | Chemin pour sauvegarder le modèle entraîné (.pkl) |
| `-load PATH` | chaîne | - | Chemin pour charger un modèle pré-entraîné (.pkl) |
| `-dontlearn` | flag | False | Désactive l'apprentissage (mode évaluation uniquement) |
| `-step-by-step` | flag | False | Active le mode pas-à-pas (pause à chaque mouvement) |

#### Options Bonus

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `-speed N` | entier | 50 | Vitesse du jeu en FPS (frames per second) |
| `-size N` | entier | 10 | Taille du plateau (N×N cellules, max 40) |
| `-stats` | flag | False | Affiche les statistiques toutes les 50 sessions |
| `-plot` | flag | False | Trace les courbes d'apprentissage (longueur max vs sessions) |

### ■ Exemples d'Utilisation

#### 1. Entraîner un modèle (sans affichage)

```bash
# Entraînement basique avec 100 sessions
python3 snake.py -sessions 100 -visual off -save models/my_model.pkl

# Entraînement avec statistiques
python3 snake.py -sessions 1000 -visual off -save models/1000sess.pkl -stats
```

#### 2. Entraîner avec affichage graphique

```bash
# Entraînement avec visualisation
python3 snake.py -sessions 50 -visual on -speed 10

# Entraînement avec visualisation et mode pas-à-pas
python3 snake.py -sessions 10 -visual on -step-by-step -speed 5
```

#### 3. Évaluer un modèle pré-entraîné

```bash
# Charger et évaluer sans apprentissage
python3 snake.py -load models/1000sess.pkl -sessions 10 -dontlearn

# Évaluation avec visualisation
python3 snake.py -load models/1000sess.pkl -sessions 5 -visual on -dontlearn -speed 10
```

#### 4. Entraîner avec différentes tailles de plateau

```bash
# Plateau 15×15
python3 snake.py -sessions 100 -size 15 -visual off -save models/15x15_100sess.pkl

# Plateau 20×20
python3 snake.py -sessions 200 -size 20 -visual off -save models/20x20_200sess.pkl
```

#### 5. Visualiser les courbes d'apprentissage

```bash
# Entraînement avec tracé des statistiques
python3 snake.py -sessions 500 -visual off -plot -stats -save models/500sess.pkl
```

#### 6. Mode terminal uniquement (sans GUI)

```bash
# Affichage terminal uniquement
python3 snake.py -sessions 50 -visual off -terminal on

# Sans affichage du tout (entraînement rapide)
python3 snake.py -sessions 1000 -visual off -terminal off
```

### ■ Règles du Jeu

- **Taille du plateau** : 10×10 cellules par défaut (configurable)
- **Pommes vertes** : 2 pommes vertes apparaissent aléatoirement
- **Pomme rouge** : 1 pomme rouge apparaît aléatoirement
- **Longueur initiale** : Le serpent commence avec une longueur de 3 cellules
- **Conditions de fin** :
  - Collision avec un mur → Game Over
  - Collision avec sa propre queue → Game Over
  - Longueur du serpent atteint 0 → Game Over
- **Mécaniques** :
  - Manger une pomme verte : longueur +1, nouvelle pomme verte apparaît
  - Manger une pomme rouge : longueur -1, nouvelle pomme rouge apparaît

---


---

## ▌Exemple de Sortie

### Entraînement

```bash
$ python3 snake.py -sessions 100 -visual off -save models/test.pkl -stats

SESSION 1 - STATISTICS:
  Max length: 3 at session 1
  Total reward: -109.00
  Steps: 10
  Epsilon: 0.0999
  Learned states: 5
  Over: True

SESSION 50 - STATISTICS:
  Max length: 4 at session 45
  Total reward: -105.00
  Steps: 8
  Epsilon: 0.0775
  Learned states: 52
  Over: True

SESSION 100 - STATISTICS:
  Max length: 4 at session 87
  Total reward: -103.00
  Steps: 4
  Epsilon: 0.0905
  Learned states: 39
  Over: True

Model saved to ./models/test.pkl
```

### Évaluation

```bash
$ python3 snake.py -load models/1000sess.pkl -sessions 5 -dontlearn -visual on

Loading model from: models/1000sess.pkl
Model loaded successfully.

Evaluation session 1/5...
Evaluation session 2/5...
Evaluation session 3/5...
Evaluation session 4/5...
Evaluation session 5/5...

Average performance: Length 7, Steps 1
```

---

## ▌Structure du Projet

```
Learn2Slither/
├── snake.py              # Script principal (point d'entrée)
├── agent.py              # Implémentation de l'agent Q-learning
├── game_data.py          # Gestion du plateau de jeu et logique
├── game_gui.py           # Interface graphique Pygame
├── requirements.txt      # Dépendances Python
├── models/               # Modèles IA pré-entraînés
│   ├── 1sess.pkl        # Modèle entraîné avec 1 session
│   ├── 10sess.pkl       # Modèle entraîné avec 10 sessions
│   ├── 100sess.pkl      # Modèle entraîné avec 100 sessions
│   ├── 1000sess.pkl     # Modèle entraîné avec 1000 sessions
│   ├── 1500sess.pkl     # Modèle entraîné avec 1500 sessions
│   ├── 2000sess.pkl     # Modèle entraîné avec 2000 sessions
│   ├── 2500sess.pkl     # Modèle entraîné avec 2500 sessions
│   ├── 3000sess.pkl     # Modèle entraîné avec 3000 sessions
│   ├── 3500sess.pkl     # Modèle entraîné avec 3500 sessions
│   ├── 4000sess.pkl     # Modèle entraîné avec 4000 sessions
│   ├── 4500sess.pkl     # Modèle entraîné avec 4500 sessions
│   └── 5000sess.pkl     # Modèle entraîné avec 5000 sessions
├── assets/               # Ressources graphiques (sprites)
├── fonts/                # Polices de caractères
└── README.md            # Ce fichier
```

---

## ▌Résultats de Performance

### ■ Statistiques d'Entraînement Réelles

Les statistiques suivantes proviennent de l'entraînement réel du modèle :

| Sessions | Longueur Max | Récompense Totale | Étapes | Epsilon | États Appris | Bonus Atteints |
|----------|--------------|-------------------|--------|---------|--------------|----------------|
| 1 | 3 | -109.00 | 10 | 0.0999 | 5 | - |
| 10 | 3 | -102.00 | 3 | 0.0990 | 14 | - |
| 100 | 4 | -103.00 | 4 | 0.0905 | 39 | - |
| 500 | 5 | -100.00 | 1 | 0.0606 | 88 | - |
| 1000 | 7 | -100.00 | 1 | 0.0368 | 114 | - |
| 1500 | 8 | -108.00 | 20 | 0.0223 | 137 | - |
| 2000 | 10 | -118.00 | 32 | 0.0135 | 167 | [10] |
| 2500 | 19 | -128.00 | 62 | 0.0100 | 201 | [10, 15] |
| 3000 | 25 | -135.00 | 69 | 0.0100 | 222 | [10, 15, 20, 25] |
| 3500 | 63 | -610.00 | 680 | 0.0100 | 262 | [10, 15, 20, 25, 30, 35] |
| 4000 | 67 | -530.00 | 664 | 0.0100 | 271 | [10, 15, 20, 25, 30, 35] |
| 4500 | 71 | -277.00 | 409 | 0.0100 | 269 | [10, 15, 20, 25, 30, 35] |
| 5000 | 65 | -556.00 | 468 | 0.0100 | 270 | [10, 15, 20, 25, 30, 35] |

### ■ Analyse des Performances

- **Progression** : Le modèle atteint une longueur maximale de **71 cellules** après 4500 sessions
- **Taux de réussite** : Le modèle développe des stratégies efficaces pour éviter les collisions
- **Exploration** : L'epsilon décroît progressivement de 0.1 à 0.01, favorisant l'exploitation
- **États appris** : Le modèle découvre environ **270 états uniques** après 5000 sessions
- **Bonus** : Le modèle atteint régulièrement les longueurs de bonus (10, 15, 20, 25, 30, 35)

---

## ▌Détails Techniques

### Architecture
Le projet suit une architecture modulaire :
- **`game_data.py`** : Gère l'état du jeu, les mouvements du serpent, le placement des pommes
- **`agent.py`** : Implémente l'algorithme Q-learning et la prise de décision
- **`game_gui.py`** : Gère l'affichage visuel et l'interaction utilisateur
- **`snake.py`** : Orchestre l'entraînement et l'évaluation

### Qualité du Code
- Respect des standards Python PEP 8
- Design modulaire avec séparation claire des responsabilités
- Gestion d'erreurs complète
- Documentation extensive avec docstrings
- Validation des arguments en ligne de commande

### Outils de Développement

Pour vérifier et formater le code :

```bash
flake8 snake.py
autopep8 --in-place --aggressive --aggressive snake.py
```

---

## ▌Evaluation

The project meets all mandatory requirements:
- ✅ Q-learning implementation
- ✅ Proper snake vision limitation
- ✅ Correct reward system
- ✅ Model save/load functionality
- ✅ Graphical interface
- ✅ Command line interface
- ✅ Required model files (1, 10, 100 sessions)

Bonus features implemented:
- 🎉 Enhanced visual interface
- 🎉 Variable board sizes
- 🎉 High performance achievements

---

## 📜 License

This project was completed as part of the **42 School** curriculum.\
It is intended for **academic purposes only** and follows the evaluation requirements set by 42.

Unauthorized public sharing or direct copying for **grading purposes** is discouraged.\
If you wish to use or study this code, please ensure it complies with **your school's policies**.

