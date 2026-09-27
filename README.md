# Learn2Slither 🐍
![Score](https://img.shields.io/badge/Score-125%25-brightgreen)  
**A snake that learns how to behave in an environment through trial and error using Q-learning reinforcement learning**

> Train an AI agent to play Snake using Q-learning algorithm with limited vision and proper reward system.

---

## ▌Project Overview

This project implements a complete Snake game with an AI agent that learns to play using **Q-learning reinforcement learning**.\
The snake learns through trial and error, gradually improving its performance over multiple training sessions.\
It serves as an introduction to **reinforcement learning fundamentals**, written from scratch without using high-level ML libraries.

📘 Educational AI project: **you'll build and train your own Q-learning model step-by-step**.

<div align="center">

| Snake in action |
|:---:|
| <img src="https://github.com/user-attachments/assets/437c5b6b-0a8c-4bdf-97e9-18bbfa76264f" alt="Snake in action" width="500"> |

</div>

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

## ▌Bonus Features

- ■ **Enhanced GUI**: Lobby system, configuration panel, statistics tracking
- ■ **Variable Board Sizes**: Support for different dimensions (10×10, 15×15, 20×20, up to 40×40)
- ■ **High Performance**: Maximum length achieved of **71 cells** after 4500 sessions
- ■ **Advanced Parameters**: Speed control (FPS), step-by-step mode, learning curve plotting
- ■ **Detailed Statistics**: Periodic performance display with achieved bonuses

> ⚠️ These features are only evaluated if the core program works flawlessly.

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
Learning Rate: 0.01
Discount Factor: 0.9
Epsilon: 0.1 (exploration rate)
Epsilon Decay: 0.999 (per session)
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
- `tabulate` (table formatting)
- `colorama` (terminal colors)
- `matplotlib` (plotting)

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

## ▌Usage Instructions

### ■ Basic Syntax

```bash
python3 snake.py [OPTIONS]
```

or directly:

```bash
./snake [OPTIONS]
```

### ■ Available Options

#### Main Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-sessions N` | integer | 10 | Number of training sessions |
| `-visual [on\|off]` | choice | off | Enable/disable graphical display (Pygame) |
| `-terminal [on\|off]` | choice | on | Enable/disable terminal game display |
| `-save PATH` | string | - | Path to save the trained model (.pkl) |
| `-load PATH` | string | - | Path to load a pre-trained model (.pkl) |
| `-dontlearn` | flag | False | Disable learning (evaluation mode only) |
| `-step-by-step` | flag | False | Enable step-by-step mode (pause at each move) |

#### Bonus Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-speed N` | integer | 50 | Game speed in FPS (frames per second) |
| `-size N` | integer | 10 | Board size (N×N cells, max 40) |
| `-stats` | flag | False | Display statistics every 50 sessions |
| `-plot` | flag | False | Plot learning curves (max length vs sessions) |

### ■ Usage Examples

#### 1. Train a model (without display)

```bash
# Basic training with 100 sessions
python3 snake.py -sessions 100 -visual off -save models/my_model.pkl

# Training with statistics
python3 snake.py -sessions 1000 -visual off -save models/1000sess.pkl -stats
```

#### 2. Train with graphical display

```bash
# Training with visualization
python3 snake.py -sessions 50 -visual on -speed 10

# Training with visualization and step-by-step mode
python3 snake.py -sessions 10 -visual on -step-by-step -speed 5
```

#### 3. Evaluate a pre-trained model

```bash
# Load and evaluate without learning
python3 snake.py -load models/1000sess.pkl -sessions 10 -dontlearn

# Evaluation with visualization
python3 snake.py -load models/1000sess.pkl -sessions 5 -visual on -dontlearn -speed 10
```

#### 4. Train with different board sizes

```bash
# 15×15 board
python3 snake.py -sessions 100 -size 15 -visual off -save models/15x15_100sess.pkl

# 20×20 board
python3 snake.py -sessions 200 -size 20 -visual off -save models/20x20_200sess.pkl
```

#### 5. Visualize learning curves

```bash
# Training with statistics plotting
python3 snake.py -sessions 500 -visual off -plot -stats -save models/500sess.pkl
```

#### 6. Terminal mode only (without GUI)

```bash
# Terminal display only
python3 snake.py -sessions 50 -visual off -terminal on

# No display at all (fast training)
python3 snake.py -sessions 1000 -visual off -terminal off
```

### ■ Game Rules

- **Board size**: 10×10 cells by default (configurable)
- **Green apples**: 2 green apples appear randomly
- **Red apple**: 1 red apple appears randomly
- **Initial length**: The snake starts with a length of 3 cells
- **End conditions**:
  - Collision with a wall → Game Over
  - Collision with own tail → Game Over
  - Snake length reaches 0 → Game Over
- **Mechanics**:
  - Eating a green apple: length +1, new green apple appears
  - Eating a red apple: length -1, new red apple appears

---


---

## ▌Example Output

### Training

```bash
$ python3 snake.py -sessions 100 -visual off -save models/test.pkl -stats

SESSION 50 - STATISTICS:
  Max length: '4' at session '18'
  Total reward: -100.00
  Steps: 1
  Epsilon: 0.0951
  Learned states: 27
  Over: True

SESSION 100 - STATISTICS:
  Max length: '4' at session '18'
  Total reward: -104.00
  Steps: 5
  Epsilon: 0.0905
  Learned states: 40
  Over: True

Model saved to models/test.pkl
```

Values vary between runs; this is one run of `-sessions 100`.

### Evaluation

```bash
$ python3 snake.py -load models/1000sess.pkl -sessions 5 -dontlearn -visual on

Model loaded from models/1000sess.pkl
Episodes trained: 1000
States in Q-table: 119
```

---

## ▌Project Structure

```
Learn2Slither/
├── snake.py              # Main script (entry point)
├── agent.py              # Q-learning agent implementation
├── game_data.py          # Game board and logic management
├── game_gui.py           # Pygame graphical interface
├── requirements.txt      # Python dependencies
├── models/               # Pre-trained AI models
│   ├── 1sess.pkl        # Model trained with 1 session
│   ├── 10sess.pkl       # Model trained with 10 sessions
│   ├── 100sess.pkl      # Model trained with 100 sessions
│   ├── 500sess.pkl      # Model trained with 500 sessions
│   ├── 1000sess.pkl     # Model trained with 1000 sessions
│   ├── 1500sess.pkl     # Model trained with 1500 sessions
│   ├── 2000sess.pkl     # Model trained with 2000 sessions
│   ├── 2500sess.pkl     # Model trained with 2500 sessions
│   ├── 3000sess.pkl     # Model trained with 3000 sessions
│   ├── 3500sess.pkl     # Model trained with 3500 sessions
│   ├── 4000sess.pkl     # Model trained with 4000 sessions
│   ├── 4500sess.pkl     # Model trained with 4500 sessions
│   └── 5000sess.pkl     # Model trained with 5000 sessions
├── assets/               # Graphical resources (sprites)
├── fonts/                # Font files
└── README.md            # This file
```

---

## ▌Performance Results

### ■ Real Training Statistics

The following statistics come from the actual model training:

| Sessions | Max Length | Total Reward | Steps | Epsilon | Learned States | Bonuses Achieved |
|----------|------------|--------------|-------|---------|----------------|------------------|
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

### ■ Performance Analysis

- **Progression**: The model reaches a maximum length of **71 cells** after 4500 sessions
- **Success rate**: The model develops effective strategies to avoid collisions
- **Exploration**: Epsilon gradually decreases from 0.1 to 0.01, favoring exploitation
- **Learned states**: The model discovers approximately **270 unique states** after 5000 sessions
- **Bonuses**: The model regularly achieves bonus lengths (10, 15, 20, 25, 30, 35)

---

## ▌Technical Details

### Architecture
The project follows a modular architecture:
- **`game_data.py`**: Manages game state, snake movements, apple placement
- **`agent.py`**: Implements Q-learning algorithm and decision making
- **`game_gui.py`**: Handles visual display and user interaction
- **`snake.py`**: Orchestrates training and evaluation

### Code Quality
- Follows Python PEP 8 standards
- Modular design with clear separation of concerns
- Comprehensive error handling
- Extensive documentation with docstrings
- Command-line argument validation

### Development Tools

To check and format the code:

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

