import sys
import random
import traceback
import numpy as np
import tensorflow as tf

from tabulate import tabulate
from collections import deque
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from game_data import Game



print("=== GPU CONFIGURATION ===")
print(f"Available GPU(s): {tf.config.list_physical_devices('GPU')}")
print(f"GPU detected: {len(tf.config.list_physical_devices('GPU')) > 0}")
if tf.config.list_physical_devices('GPU'):
    print("✅ GPU will be used for training")
else:
    print("❌ No GPU detected, using CPU")
print("=========================")

TAU = 1e-3
NUM_STEPS_FOR_UPDATE = 4
MINIBATCH_SIZE = 64

class DeepQNetwork:

    EMPTY = 0
    WALL = 1
    HEAD = 2
    BODY = 3
    RED_APPLE = 4
    GREEN_APPLE = 5

    ACTIONS = ['up', 'down', 'left', 'right']

    CHARS = [
        'EMPTY',
        'WALL',
        'HEAD',
        'BODY',
        'RED_APPLE',
        'GREEN_APPLE',
    ]

    def __init__(self, game: Game):
        self.gamma = 0.95
        self.alpha = 5e-3
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory_size = 100_000
        self.minibatch_size = MINIBATCH_SIZE
        self.warmup_steps = 500

        self.q_network = None
        self.target_q_network = None
        self.optimizer = None
        self.loss_fn = keras.losses.MeanSquaredError()

        self.game = game
        self.experience = deque(maxlen=self.memory_size)

        self.states = None
        self.state_size = None
        self.num_actions = len(self.ACTIONS)

        self.train_step_count = 0
        self.duration = 0
        self.step_count = 0
        self.is_over = False
        self.last_action = None

        self.max_length_global = 0
        self.max_duration_global = 0
        self.learning_mode = True



    def ft_define_initial_model(self):
        self.q_network = Sequential([
            Input(shape=self.state_size),
            Dense(128, activation='relu', name='layer1'),
            Dense(128, activation='relu', name='layer2'),
            Dense(self.num_actions, activation='linear'),
        ])
        self.target_q_network = Sequential([
            Input(shape=self.state_size),
            Dense(128, activation='relu', name='layer1'),
            Dense(128, activation='relu', name='layer2'),
            Dense(self.num_actions, activation='linear'),
        ])

        self.optimizer = Adam(learning_rate=self.alpha)
        self.q_network.compile(optimizer=self.optimizer, loss=self.loss_fn)
        self.target_q_network.compile(optimizer=self.optimizer, loss=self.loss_fn)

        sym = keras.Input(shape=self.state_size, name="build_input")
        _ = self.q_network(sym)
        _ = self.target_q_network(sym)

        # try:
        #     self.q_network.load_weights('./models/DQN/50sess.weights.h5')
        #     print("[info] Loaded weights from ./models/DQN/50sess.weights.h5")
        # except Exception:
        #     print("[info] No existing weights found, starting fresh.")


        self.target_q_network.set_weights(self.q_network.get_weights())



    # def ft_choose_action(self, state_vec):
    #     q_values = self.q_network.predict(state_vec[None, :], verbose=0)[0]

    #     u = random.uniform(0, 1)
    #     if u < self.epsilon:
    #         action = random.randrange(self.num_actions)
    #     else:
    #         action = int(np.argmax(q_values))

    #     self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)  

    #     return action

    def ft_get_safe_actions(self, vision: str):    
        safe_actions = []
        directions = ['up', 'down', 'left', 'right']
        
        for i, direction in enumerate(directions):
            if vision[i] not in ['W', 'S']:
                safe_actions.append(i)
        
        return safe_actions


    def ft_encode_vision(self, vision: str) -> np.ndarray:
        char_to_num = {
            '0': 0.0,
            'W': 1.0,
            'H': 2.0,
            'S': 3.0,
            'G': 4.0,
            'R': 5.0
        }
        state_vec = np.array([], dtype=np.float32)
        for char in vision:
            state_vec = np.append(state_vec, char_to_num.get(char, 0.0))
        return state_vec

    def ft_choose_action(self, states: np.ndarray, vision: str) -> int:
        u = random.random()
        
        q_values = self.q_network.predict(states[None, :], verbose=0)[0]
        safe_actions = self.ft_get_safe_actions(vision)
        
        if not safe_actions:
            safe_actions = list(range(len(self.ACTIONS)))
        
        if self.learning_mode and u < self.epsilon:
            action_index = random.choice(safe_actions)
        else:
            safe_q_values = [q_values[i] for i in safe_actions]
            best_action_index = safe_actions[np.argmax(safe_q_values)]
            action_index = best_action_index
        
        self.last_action = action_index
        return action_index

    def ft_decode_vision(self, state_vec: np.ndarray) -> str:
        """Convertir le vecteur d'état en vision string"""
        num_to_char = {
            0.0: '0',  # EMPTY
            1.0: 'W',  # WALL
            2.0: 'H',  # HEAD
            3.0: 'S',  # SNAKE_BODY
            4.0: 'G',  # GREEN_APPLE
            5.0: 'R'   # RED_APPLE
        }
        
        vision = ""
        for val in state_vec:
            vision += num_to_char.get(val, '0')
        
        return vision    

    def ft_store_experience(self, state, action_index, reward, next_state, is_over):
        self.experience.append((
            np.array(state, dtype=np.float32, copy=True),
            int(action_index),
            float(reward),
            np.array(next_state, dtype=np.float32, copy=True),
            bool(is_over),
        ))

    def ft_calculate_reward(self, code: int, is_over: bool) -> float:
        if is_over:
            return -100.0
        elif code == Game.GREEN_APPLE:
            return +10.0
        elif code == Game.RED_APPLE:
            return -10.0
        else:
            return -1.0 

    def ft_compute_loss(self):
        batch = random.sample(self.experience, k=MINIBATCH_SIZE)

        states      = np.array([exp[0] for exp in batch], dtype=np.float32)
        actions     = np.array([exp[1] for exp in batch], dtype=np.int32)
        rewards     = np.array([exp[2] for exp in batch], dtype=np.float32)
        next_states = np.array([exp[3] for exp in batch], dtype=np.float32)
        dones       = np.array([float(exp[4]) for exp in batch], dtype=np.float32)

        q_next_target = self.target_q_network(next_states, training=False)
        max_qsa = tf.reduce_max(q_next_target, axis=1)

        y_targets = rewards + (self.gamma * max_qsa * (1.0 - dones))
        y_targets = tf.stop_gradient(y_targets)

        q_values_all = self.q_network(states, training=True)
        idx = tf.stack([tf.range(tf.shape(q_values_all)[0]),
                        tf.cast(actions, tf.int32)], axis=1)
        q_values = tf.gather_nd(q_values_all, idx)

        loss = self.loss_fn(y_targets, q_values)

        return loss, q_values, y_targets

    def ft_update_target_soft_update(self):
        for target_weights, q_net_weights in zip(
            self.target_q_network.weights, self.q_network.weights
        ):
            target_weights.assign(TAU * q_net_weights + (1.0 - TAU) * target_weights)

    def ft_agent_learn(self):
        if len(self.experience) < max(MINIBATCH_SIZE, self.warmup_steps):
            return None

        self.train_step_count += 1

        with tf.GradientTape() as tape:
            loss, q_values, y_targets = self.ft_compute_loss()

        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        grad_norm = tf.linalg.global_norm([g for g in gradients if g is not None])
        
        # print(f"[train:{self.train_step_count}] | max_len={self.max_length_global} | max_duration={self.max_duration_global} | session={self.session_id} | epsilon={self.epsilon:.3f} | loss={float(loss):.2f} | grad_norm={float(grad_norm):.2f}")

        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
        self.ft_update_target_soft_update()
        return float(loss)

    def ft_check_update_conditions(self, step):
        return (step + 1) % NUM_STEPS_FOR_UPDATE == 0 and len(self.experience) >= MINIBATCH_SIZE


    def ft_initialize_model(self):
        self.experience = deque(maxlen=self.memory_size)
        initial_vision = self.game.ft_get_vision_snake()
        self.states = self.ft_encode_vision(initial_vision)
        self.state_size = (self.states.shape[0],)
        self.ft_define_initial_model()
        self.game.ft_reset()
        self.loss = 0.0

    def ft_interaction(self, sessions: int):
        self.ft_initialize_model()
        self.step_count = 0
        self.session_id = 0
        max_length_global = 0
        max_duration_global = 0
        max_num_timesteps = 1000

        for session_id in range(1, sessions + 1):
            self.game.ft_reset()
            self.session_id = session_id
            print(f"Session {session_id} started")
            self.is_over = False
            self.last_action = None
            self.step = 0

            try:
                while not self.game.is_over and self.step < max_num_timesteps:           
                    
                    vision = self.game.ft_get_vision_snake()
                    self.states = self.ft_encode_vision(vision)
                    action_index = self.ft_choose_action(self.states, vision)

                    action_str = self.ACTIONS[action_index]

                    code, is_over = self.game.ft_move_snake(action_str, "off")
                    reward = self.ft_calculate_reward(code, is_over)
                    next_state = self.ft_encode_vision(self.game.ft_get_vision_snake())
                    
                    self.ft_store_experience(self.states, action_index, reward, next_state, is_over)

                    if self.learning_mode and self.ft_check_update_conditions(self.step):
                        loss_result = self.ft_agent_learn()
                        self.loss = loss_result if loss_result is not None else 0.0
                    else:
                        self.loss = 0.0

                    self.states = next_state.copy()
                    self.is_over = is_over
    
                    len_after = len(self.game.snake)
                    if len_after > max_length_global:
                        max_length_global = len_after
                    self.max_length_global = max_length_global

                    self.step += 1

            except Exception as e:
                print("⚠️  Exception during the session !")
                print(traceback.format_exc())
                self.is_over = True
                sys.exit(1)

            if self.step > max_duration_global:
                max_duration_global = self.step
                self.max_duration_global = max_duration_global

            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


            if session_id % 10 == 0:
                self.q_network.save_weights(f'./models/DQN/{session_id}sess.weights.h5')

            loss_display = self.loss if self.loss is not None else 0.0
            print(f"[train:{self.train_step_count}] | max_len={self.max_length_global} | max_duration={self.max_duration_global} | session={self.session_id} | epsilon={self.epsilon:.3f} | loss={loss_display:.2f}")


        return max_length_global, max_duration_global
       
    def set_learning_mode(self, learning: bool) -> None:
        self.learning_mode = learning

    def __str__(self):
        return 'TEST'
    

    gamma: float
    alpha: float
    epsilon: float
    epsilon_min: float
    epsilon_decay: float
    memory_size: int
    minibatch_size: int
    warmup_steps: int
    q_network: Sequential | None
    target_q_network: Sequential | None
    optimizer: Adam | None
    loss_fn: keras.losses.Loss
    game: Game
    experience: deque
    states: np.ndarray | None
    state_size: tuple | None
    num_actions: int
    train_step_count: int
    step: int
    step_count: int
    is_over: bool
    last_action: int | None
    max_length_global: int
    max_duration_global: int


def main():
    size = 10
    game = Game(size, size)
    networks = DeepQNetwork(game)
    try:
        sessions = 100
        max_length, max_duration = networks.ft_interaction(sessions)
        print(f"Game over, max length: {max_length}, max_duration: {max_duration}")
        # networks.q_network.save_weights('./standard.weights.h5')
    except KeyboardInterrupt:
        del game
        del networks
        sys.exit(1)

if __name__ == "__main__":
    main()
