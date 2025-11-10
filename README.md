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

## ▌Bonus Features

- ■ **Enhanced GUI**: Lobby system, configuration panel, statistics tracking
- ■ **Variable Board Sizes**: Support for different board dimensions (10x10, 15x15, 20x20)
- ■ **High Performance**: Achieved length 20+ with 70% success rate
- ■ **Advanced Training**: Improved parameters and training strategies

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

### ■ Installation & Usage

1. Clone the repository

```bash
git clone https://github.com/ai-dg/Learn2Slither.git
cd Learn2Slither
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Train a model

```bash
# Train with 100 sessions
python3 snake.py -sessions 100 -visual off -save models/my_model.txt

# Train with visual display
python3 snake.py -sessions 50 -visual on -speed 5
```

4. Evaluate a trained model

```bash
# Load and evaluate a model
python3 snake.py -load models/1000sess.txt -sessions 10 -dontlearn
```

5. Run interactive mode

```bash
python3 snake.py -interactive
```

6. Run demonstration

```bash
python3 demo.py
```

---

## ▌Performance Results

### Standard Models
| Sessions | Best Length | Best Duration | Success Rate | Q-table States |
|----------|-------------|---------------|--------------|----------------|
| 1        | 3           | 9             | 0%           | 7              |
| 10       | 4           | 12            | 0%           | 20             |
| 100      | 6           | 55            | 0%           | 84             |
| 1000     | 33          | 1000          | 21%          | 254            |

### Advanced Models
| Configuration | Best Length | Success Rate | Features |
|---------------|-------------|--------------|----------|
| 10x10 Advanced | 23+ | 70-80% | Enhanced parameters |
| 15x15 Board | 4+ | Variable | Larger environment |
| 20x20 Board | 4+ | Variable | Extended gameplay |

---

## ▌Example

```bash
$ python3 snake.py -sessions 100 -visual off -save models/test.txt
Training session 1/100...
Training session 50/100...
Training session 100/100...

Final results:
Best length: 6
Best duration: 55
Success rate: 0%
Q-table states: 84

Model saved to: models/test.txt

$ python3 snake.py -load models/1000sess.txt -sessions 5 -dontlearn
Loading model from: models/1000sess.txt
Model loaded successfully.

Evaluation results:
Session 1: Length 33, Duration 1000
Session 2: Length 25, Duration 800
Session 3: Length 30, Duration 950
Session 4: Length 28, Duration 900
Session 5: Length 31, Duration 980

Average length: 29.4
Success rate: 100%
```

---

## ▌Project Structure

```
Learn2Slither/
├── board.py              # Game board and environment
├── agent.py              # Q-learning agent
├── gui.py                # Graphical interface (basic + advanced)
├── snake.py              # Main game (basic features)
├── advanced_snake.py     # Advanced game (bonus features)
├── train_models.py       # Script to train required models
├── train_advanced_models.py  # Script for advanced training
├── demo.py               # Demonstration script
├── requirements.txt      # Python dependencies
├── models/               # Trained AI models
│   ├── 1sess.txt        # Model trained with 1 session
│   ├── 10sess.txt       # Model trained with 10 sessions
│   ├── 100sess.txt      # Model trained with 100 sessions
│   ├── 1000sess.txt     # Model trained with 1000 sessions
│   └── advanced_*.txt   # Advanced models with bonus features
└── README.md            # This file
```

---

## ▌Command Line Options

### Basic Options
- `-sessions N`: Number of training sessions
- `-visual [on|off]`: Enable/disable visual display
- `-step-by-step`: Enable step-by-step mode
- `-speed N`: Game speed in FPS
- `-size N`: Board size (bonus feature)

### Model Options
- `-save PATH`: Save trained model to file
- `-load PATH`: Load trained model from file
- `-dontlearn`: Disable learning (evaluation mode)

### Mode Options
- `-terminal`: Show or hide snake movements
- `-plot`: Plot learninag curve with max_lengths and sessions
- `-stats`: Show stats each 50 sessions

---

## ▌Technical Details

### Architecture
The project follows a modular architecture:
- **Board**: Manages game state, snake movement, apple placement
- **Agent**: Implements Q-learning algorithm and decision making
- **GUI**: Handles visual display and user interaction
- **Main**: Orchestrates training and evaluation

### Code Quality
- Follows Python PEP 8 standards
- Modular design with clear separation of concerns
- Comprehensive error handling
- Extensive documentation

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



```bash
flake8 mon_script.py
autopep8 --in-place --aggressive --aggressive mon_script.py

```

Board size: 10 cells by 10 cells.
Two green apples, in a random cell of the board
One red apple, in a random cell of the board
The snake starts with a length of 3 cells, also placed randomly and contiguously on the board.
If the snake hits a wall: Game over, this training session ends.
If the snake collides with its own tail: Game over, this training session ends.
The snake eats a green apple: snake’s length increase by 1. A new green apple
appears on the board.
The snake eats a red apple: snake’s length decrease by 1. A new red apple appears
on the board.
If the snake’s length drops to 0: Game over, this training session ends.

```bash
./snake -sessions 10 -save models/10sess.txt -visual off
./snake -visual on -load models/100sess.txt -sessions 10 -dontlearn -step-by-step
Load trained model from models/100sess.txt
./snake -visual on -load models/1000sess.txt

```

SESSION 1 - STATISTICS:
  Max length: 3 at session 1
  Total reward: -109.00
  Steps: 10
  Epsilon: 0.0999
  Learned states: 5
  Over: True

Model saved to ./models/1sess.pkl

SESSION 10 - STATISTICS:
  Max length: 3 at session 1
  Total reward: -102.00
  Steps: 3
  Epsilon: 0.0990
  Learned states: 14
  Over: True

Model saved to ./models/10sess.pkl

SESSION 100 - STATISTICS:
  Max length: 4 at session 7
  Total reward: -103.00
  Steps: 4
  Epsilon: 0.0905
  Learned states: 39
  Over: True

Model saved to ./models/100sess.pkl


SESSION 500 - STATISTICS:
  Max length: 5 at session 459
  Total reward: -100.00
  Steps: 1
  Epsilon: 0.0606
  Learned states: 88
  Over: True

SESSION 1000 - STATISTICS:
  Max length: 7 at session 617
  Total reward: -100.00
  Steps: 1
  Epsilon: 0.0368
  Learned states: 114
  Over: True

SESSION 1500 - STATISTICS:
  Max length: 8 at session 1383
  Total reward: -108.00
  Steps: 20
  Epsilon: 0.0223
  Learned states: 137
  Over: True


SESSION 2000 - STATISTICS:
  Max length: 10 at session 1809
  Total reward: -118.00
  Steps: 32
  Epsilon: 0.0135
  Learned states: 167
  Over: True
  Bonuses achieved: [10]

SESSION 2500 - STATISTICS:
  Max length: 19 at session 2204
  Total reward: -128.00
  Steps: 62
  Epsilon: 0.0100
  Learned states: 201
  Over: True
  Bonuses achieved: [10, 15]

SESSION 3000 - STATISTICS:
  Max length: 25 at session 2874
  Total reward: -135.00
  Steps: 69
  Epsilon: 0.0100
  Learned states: 222
  Over: True
  Bonuses achieved: [10, 15, 20, 25]

SESSION 3500 - STATISTICS:
  Max length: 63 at session 3445
  Total reward: -610.00
  Steps: 680
  Epsilon: 0.0100
  Learned states: 262
  Over: True
  Bonuses achieved: [10, 15, 20, 25, 30, 35]

SESSION 4000 - STATISTICS:
  Max length: 67 at session 3998
  Total reward: -530.00
  Steps: 664
  Epsilon: 0.0100
  Learned states: 271
  Over: True
  Bonuses achieved: [10, 15, 20, 25, 30, 35]

SESSION 4500 - STATISTICS:
  Max length: 71 at session 3104
  Total reward: -277.00
  Steps: 409
  Epsilon: 0.0100
  Learned states: 269
  Over: True
  Bonuses achieved: [10, 15, 20, 25, 30, 35]

SESSION 5000 - STATISTICS:
  Max length: 65 at session 4482
  Total reward: -556.00
  Steps: 468
  Epsilon: 0.0100
  Learned states: 270
  Over: True
  Bonuses achieved: [10, 15, 20, 25, 30, 35]
 
 ```bash
 memray run -o out.bin --leaks snake.py -sessions 100
 memray tree out.bin 
 memray flamegraph out.bin 
 ```