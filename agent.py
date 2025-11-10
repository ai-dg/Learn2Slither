#!/usr/bin/env python3

import sys
import time
import random
import pickle
from game_data import Game
from game_gui import GameGUI
from collections import deque
from typing import Dict, List, Set, Deque
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')


class QLearning:
    """
    Logic:
    - Encapsulate the Q-learning agent and its training workflow.

    Return:
    - None
    """
    # ---------------------------------------------------------------------------
    # Class-Level Configuration

    ACTIONS = [
        'up',
        'down',
        'left',
        'right'
    ]

    CHAR_DEFINITION = [
        'EMPTY',
        'WALL',
        'HEAD',
        'BODY',
        'RED_APPLE',
        'GREEN_APPLE',
    ]

    # ---------------------------------------------------------------------------
    # Initialization

    def ft_initialize_variables(self) -> None:
        """
        Logic:
        - Reset runtime trackers for a fresh training session.

        Return:
        - None
        """
        self.q_table = {}
        self.learning_mode = True
        self.episodes_trained = 0
        self.total_rewards = []
        self.last_action = 'up'
        self.bonus_achieved = set()
        self.total_reward = 0
        self.steps = 0
        self.timesteps_reward = 0
        self.max_length = 0
        self.visual_mode = False
        self.gui = None
        self.step_by_step_mode = False
        self.gui_data = None
        self.max_length = 0
        self.max_length_session = 0
        self.if_max_length = False
        self.session_id = 0
        self.q_value_taken = 0.0
        self.green_apples_taken = 0
        self.red_apples_taken = 0
        self.game_over_reason = 'None'
        self.is_over = False
        self.code_reason = ""
        self.load_path = "None"
        self.terminal_mode = 'off'
        self.show_stats = False
        self.plot = False

    def ft_initialize_values_for_gui(self):
        """
        Logic:
        - Build the default payload exposed to the GUI overlay.

        Return:
        - None
        """
        self.speed_gui = 50

        self.gui_data = {
            'next_step': False,
            'speed': self.speed_gui,
            'learn': self.learning_mode,
            'step_by_step': self.step_by_step_mode,
            'session_id': 0,
            'max_length': 0,
            'epsilon': self.epsilon,
            'q_value_taken': 0,
            'green_apples_taken': 0,
            'red_apples_taken': 0,
            'game_over_reason': 'None',
            'game_over': False,
            'model_loaded': 'None',
            'duration': 0,
            'reward_session': 0,
            'vision': ['', '', '', ''],
        }

    def __init__(
            self,
            game: Game,
            learning_rate: float = 0.01,
            gamma: float = 0.9,
            epsilon: float = 0.1,
            epsilon_decay: float = 0.999,
            epsilon_min: float = 0.01) -> None:
        """
        Logic:
        - Store hyperparameters and bootstrap agent state.

        Return:
        - None
        """
        self.game = game
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.ft_initialize_variables()
        self.ft_initialize_values_for_gui()

    # ---------------------------------------------------------------------------
    # Q-Value Updates

    def ft_update_q_value(
            self,
            vision: str,
            action: str,
            reward: float,
            next_vision: str,
            done: bool) -> None:
        """
        Logic:
        - Apply the temporal-difference update for the transition.

        Return:
        - None
        """
        if not self.learning_mode:
            return

        current_q = self.q_table[vision][action]

        if done:
            target_q = reward
        else:
            if next_vision not in self.q_table:
                self.q_table[next_vision] = self.ft_initialize_vision_q_table(
                    next_vision)

            max_next_q = max(self.q_table[next_vision].values())
            target_q = reward + self.gamma * max_next_q

        new_q = current_q + self.learning_rate * (target_q - current_q)
        self.q_table[vision][action] = new_q

    # ---------------------------------------------------------------------------
    # Progress Tracking

    def ft_check_bonus(self, session: int, bonus_interval: int) -> None:
        """
        Logic:
        - Print milestone statistics while the agent trains.

        Return:
        - None
        """
        bonus_levels = [10, 15, 20, 25, 30, 35]

        for bonus in bonus_levels:
            if self.max_length >= bonus and bonus not in self.bonus_achieved:
                print(
                    f"BONUS ACHIEVED: Length "
                    f"'{bonus}' reached at session '{session}'")
                print(f"Current maximum length: {self.max_length}")
                self.bonus_achieved.add(bonus)

        if session % bonus_interval == 0:
            print(f"\nSESSION {session} - STATISTICS:")
            print(
                f"  Max length: "
                f"'{self.max_length}' at session "
                f"'{self.max_length_session}'")
            print(f"  Total reward: {self.timesteps_reward:.2f}")
            print(f"  Steps: {self.steps}")
            print(f"  Epsilon: {self.epsilon:.4f}")
            print(f"  Learned states: {len(self.q_table)}")
            print(f"  Over: {self.game.is_over}")

            if self.bonus_achieved:
                print(f"  Bonuses achieved: {sorted(self.bonus_achieved)}")
            print()

    # ---------------------------------------------------------------------------
    # Action Selection

    def ft_initialize_vision_q_table(self, vision: str) -> Dict[str, float]:
        """
        Logic:
        - Create zero-initialised action values for a new vision.

        Return:
        - Dict[str, float]
        """
        q_table = {}
        q_table[vision] = {}
        for action in self.ACTIONS:
            q_table[vision][action] = 0.0
        return q_table[vision]

    def ft_choose_action(self, vision: str) -> str:
        """
        Logic:
        - Select an action via ε-greedy policy on the current vision.

        Return:
        - str
        """
        u = random.uniform(0.01, 0.1)

        if vision not in self.q_table:
            self.q_table[vision] = self.ft_initialize_vision_q_table(vision)

        if self.learning_mode and u < self.epsilon:
            action = random.choice(self.ACTIONS)
        else:
            q_values = self.q_table[vision]
            max_q_value = max(q_values.values())
            self.q_value_taken = max_q_value
            best_actions = []
            for action, q_value in q_values.items():
                if q_value == max_q_value:
                    best_actions.append(action)
            action = random.choice(best_actions)

        self.last_action = action
        return action

    # ---------------------------------------------------------------------------
    # Reward Strategy

    def ft_calculate_reward(self, code: int, is_over: bool) -> float:
        """
        Logic:
        - Compute the scalar reward for the observed tile result.

        Return:
        - float
        """
        if is_over:
            return -100.0
        elif code == Game.GREEN_APPLE:
            return +10.0
        elif code == Game.RED_APPLE:
            return -10.0
        else:
            return -1.0

    # ---------------------------------------------------------------------------
    # Session Lifecycle

    def ft_session_cycle_traitement(self, session):
        """
        Logic:
        - Close out a training episode and persist aggregated metrics.

        Return:
        - None
        """
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        self.episodes_trained += 1
        self.total_rewards.append(self.timesteps_reward)
        if self.show_stats:
            self.ft_check_bonus(session, 50)
        if self.plot:
            self.ft_save_values_for_plot()

    # ---------------------------------------------------------------------------
    # Plotting Support

    def ft_save_values_for_plot(self) -> None:
        """
        Logic:
        - Persist the latest max-length statistics for plotting.

        Return:
        - None
        """
        if self.if_max_length:
            self.max_lengths.append(self.max_length)
            self.sessions.append(self.max_length_session)

    def ft_initialize_values_for_plot(self, sessions: int):
        """
        Logic:
        - Allocate bounded deques used to track stats across sessions.

        Return:
        - None
        """
        self.max_lengths = deque(maxlen=sessions)
        self.sessions = deque(maxlen=sessions)

    def ft_put_values_to_plot(self):
        """
        Logic:
        - Render the max-length evolution chart.

        Return:
        - None
        """
        fig, ax = plt.subplots()

        ax.plot(self.sessions, self.max_lengths, marker='o',
                linewidth=2, markersize=8, color='#2E86AB',
                markerfacecolor='#A23B72', markeredgecolor='#F18F01',
                markeredgewidth=2, label='Max Length')

        for x, y in zip(self.sessions, self.max_lengths):
            ax.annotate(f'({x}, {y})', (x, y),
                        textcoords="offset points", xytext=(0, 10),
                        ha='center', fontsize=14,
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='#F8F9FA',
                                  edgecolor='#6C757D', alpha=0.7))

        ax.set_title('Max Lengths vs Sessions - Q-Learning Snake Agent',
                     fontsize=20, fontweight='bold', pad=20)
        ax.set_xlabel('Sessions', fontsize=16, fontweight='bold')
        ax.set_ylabel('Max Lengths', fontsize=16, fontweight='bold')

        ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.5, color='gray')

        ax.legend(loc='best', fontsize=14, framealpha=0.9)

        ax.set_facecolor('#FFFFFF')
        plt.tight_layout()
        plt.show()
        plt.clf()
        plt.close('all')

    # ---------------------------------------------------------------------------
    # GUI Runtime Loop

    def ft_gui_cycle_traitement(self):
        """
        Logic:
        - Synchronise GUI interactions and honour step pacing.

        Return:
        - None
        """
        if self.visual_mode:
            self.gui_data = self.gui.ft_run_game_with_agent(self.gui_data)
            if self.gui_data['step_by_step']:
                while not self.gui_data['next_step']:
                    time.sleep(0.01)
                    self.gui_data = self.gui.ft_run_game_with_agent(
                        self.gui_data)
                    if not self.gui_data['step_by_step']:
                        break
                self.gui_data['next_step'] = False
            if self.gui_data['learn']:
                self.learning_mode = True
            else:
                self.learning_mode = False

    # ---------------------------------------------------------------------------
    # GUI Data Updates

    def ft_gui_data_update(self):
        """
        Logic:
        - Push the latest training metrics into the GUI payload.

        Return:
        - None
        """
        self.gui_data['session_id'] = self.session_id
        self.gui_data['max_length'] = self.max_length
        self.gui_data['epsilon'] = self.epsilon
        self.gui_data['q_value_taken'] = self.q_value_taken
        self.gui_data['green_apples_taken'] = self.green_apples_taken
        self.gui_data['red_apples_taken'] = self.red_apples_taken
        self.gui_data['game_over_reason'] = self.code_reason
        self.gui_data['game_over'] = self.is_over
        self.gui_data['model_loaded'] = self.load_path
        self.gui_data['duration'] = self.steps
        self.gui_data['reward_session'] = self.timesteps_reward
        self.gui_data['vision'] = self.game.ft_get_vision_snake_extended()

    # -----------------------------------------------------------------------
    # Training Loop

    def ft_train_agent(self, sessions: int = 1000) -> None:
        """
        Logic:
        - Train the agent for the requested number of sessions.

        Return:
        - None
        """
        self.game.ft_reset()
        self.ft_initialize_values_for_plot(sessions)
        max_timesteps = 1000

        for session in range(1, sessions + 1):
            self.game.ft_reset()
            self.steps = 0
            self.timesteps_reward = 0
            self.session_id = session
            self.green_apples_taken = 0
            self.red_apples_taken = 0
            self.q_value_taken = 0.0

            while not self.game.is_over and self.steps < max_timesteps:

                self.ft_gui_cycle_traitement()

                vision = self.game.ft_get_vision_snake()
                action = self.ft_choose_action(vision)

                code, is_over = self.game.ft_move_snake(
                    action, self.terminal_mode)
                self.code_reason = self.CHAR_DEFINITION[code]
                if code == Game.GREEN_APPLE:
                    self.green_apples_taken += 1
                elif code == Game.RED_APPLE:
                    self.red_apples_taken += 1
                self.is_over = is_over

                reward = self.ft_calculate_reward(code, is_over)
                self.timesteps_reward += reward
                self.total_reward += reward

                next_vision = self.game.ft_get_vision_snake()

                self.ft_update_q_value(
                    vision, action, reward, next_vision, is_over)

                current_length = len(self.game.snake)
                if current_length > self.max_length:
                    self.max_length = current_length
                    self.max_length_session = session
                    self.if_max_length = True

                self.steps += 1
                self.ft_gui_data_update()

            self.ft_session_cycle_traitement(session)

        if self.plot:
            self.ft_put_values_to_plot()

    # -----------------------------------------------------------------------
    # Configuration Hooks

    def ft_set_visual_mode(self, visual: bool, gui: GameGUI) -> None:
        """
        Logic:
        - Attach the GUI and optionally enable live rendering.

        Return:
        - None
        """
        self.visual_mode = visual
        self.gui = gui
        self.gui.ft_initialize_game()

    def ft_set_learning_mode(self, learning: bool) -> None:
        """
        Logic:
        - Enable or disable exploration updates at runtime.

        Return:
        - None
        """
        self.learning_mode = learning
        self.gui_data['learn'] = learning

    def ft_set_step_by_step_mode(self, step_by_step: bool) -> None:
        """
        Logic:
        - Toggle manual pacing between agent decisions.

        Return:
        - None
        """
        self.step_by_step_mode = step_by_step
        self.gui_data['step_by_step'] = step_by_step

    def ft_set_terminal_mode(self, terminal: str) -> None:
        """
        Logic:
        - Configure whether terminal debug output is emitted.

        Return:
        - None
        """
        self.terminal_mode = terminal

    def ft_set_speed_gui(self, speed: int) -> None:
        """
        Logic:
        - Override the GUI refresh rate.

        Return:
        - None
        """
        self.speed_gui = speed
        self.gui_data['speed'] = speed

    def ft_show_stats(self, show: bool) -> None:
        """
        Logic:
        - Toggle periodic logging of aggregate statistics.

        Return:
        - None
        """
        self.show_stats = show

    def ft_set_plot(self, plot: bool) -> None:
        """
        Logic:
        - Enable chart generation at the end of training.

        Return:
        - None
        """
        self.plot = plot

    # -----------------------------------------------------------------------
    # Model Persistence

    def ft_save_model(self, filepath: str) -> None:
        """
        Logic:
        - Serialise the Q-table and metadata to disk.

        Return:
        - None
        """
        try:
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
            self.save_path = filepath
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
        except Exception as e:
            print(f"Error saving model: {e}")
            sys.exit(1)
        print(f"Model saved to {filepath}")

    def ft_load_model(self, filepath: str):
        """
        Logic:
        - Restore a previously saved Q-table checkpoint.

        Return:
        - None
        """
        try:
            self.load_path = filepath
            self.gui_data['model_loaded'] = filepath
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            self.q_table = model_data['q_table']
            self.learning_rate = model_data.get(
                'learning_rate', self.learning_rate)
            self.gamma = model_data.get('gamma', self.gamma)
            self.epsilon = model_data.get('epsilon', self.epsilon)
            self.epsilon_decay = model_data.get(
                'epsilon_decay', self.epsilon_decay)
            self.epsilon_min = model_data.get('epsilon_min', self.epsilon_min)
            self.episodes_trained = model_data.get('episodes_trained', 0)
            self.total_rewards = model_data.get('total_rewards', [])

            print(f"Model loaded from {filepath}")
            print(f"Episodes trained: {self.episodes_trained}")
            print(f"States in Q-table: {len(self.q_table)}")

        except FileNotFoundError:
            print(f"Model file {filepath} not found")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

    # -----------------------------------------------------------------------
    # Type Hints

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
    bonus_achieved: Set[int]
    total_reward: float
    steps: int
    timesteps_reward: float
    max_length: int
    max_length_session: int
    max_lengths: Deque[int]
    sessions: Deque[int]
    if_max_length: bool


def main():
    size = 10
    windows_size = 1500
    game = Game(size, size)
    gui = GameGUI(game, windows_size)
    agent = QLearning(game)
    agent.ft_set_visual_mode(True, gui)
    sessions = 5000

    # agent.load_model('./models/1500sess.txt')
    # agent.ft_set_learning_mode(False)
    # agent.ft_set_step_by_step_mode(False)
    agent.ft_load_model('./models/5000sess.pkl')
    agent.ft_train_agent(sessions)


if __name__ == "__main__":
    main()
