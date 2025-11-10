import numpy as np
import tensorflow as tf
import keras
from keras.layers import Dense, Input
from keras.losses import MeanSquaredError as MSE
from keras.models import Sequential
from keras.optimizers import Adam
from game_data import Game
from collections import deque, namedtuple
import random
from tabulate import tabulate
tf.config.set_visible_devices([], 'GPU')


class DeepQNetwork:

    EMPTY = 0
    WALL = 1
    HEAD = 2
    BODY = 3
    RED_APPLE = 4
    GREEN_APPLE = 5

    ACTIONS = [
        'up',
        'down',
        'left',
        'right'
    ]

    CHARS = [
        'EMPTY',
        'WALL',
        'HEAD',
        'BODY',
        'RED_APPLE',
        'GREEN_APPLE',
    ]

    states : np.ndarray
    state_size : tuple
    num_actions : int
    gamma : float
    alpha : float
    epsilon : float
    epsilon_min : float
    epsilon_decay : float
    minibatch_size : int
    memory_size : int
    game : Game
    experience : deque

    # q_network : keras.models.Sequential


    def ft_encode_vision(self, vision):
        parts_state = []
        for a in self.ACTIONS:
            parts_state.extend(vision[a])


        states = np.asarray(parts_state, np.float32) / 5
        return states

    def __init__(self, game : Game):    
        # self.states = None
        # self.state_size = (self.states.shape[0],)
        # self.num_actions = len(self.ACTIONS)
        # Discount rate
        self.gamma = 0.995
        # Learning rate
        self.alpha = 1e-3
        self.epsilon = 1
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.memory_size = 100_000
        self.num_steps_for_update = 200

        self.q_network = None
        self.target_q_network = None

        
        self.game = game

        self.last_action = None  # index
        self.minibatch_size = 32
        self.warmup_steps = 100 


        # print(f"States: {self.states}")
        # print(f"State size: {self.state_size}")
        [...]


    def ft_define_initial_model(self):
        
        self.q_network = Sequential([
            Input(shape=self.state_size),
            Dense(units=64, activation='relu', name='layer1'),
            Dense(units=64, activation='relu', name='layer2'),
            Dense(units=self.num_actions, activation='linear'),
        ])

        self.target_q_network = Sequential([
            Input(shape=self.state_size),
            Dense(units=64, activation='relu', name='layer1'),
            Dense(units=64, activation='relu', name='layer2'),
            Dense(units=self.num_actions, activation='linear'),
        ])


        optimizer = Adam(learning_rate=self.alpha)
        
        # Compile Glorot Xavier (small random values)
        self.q_network.compile(
            optimizer=optimizer,
            loss=keras.losses.Huber(),
        )
        self.target_q_network.compile(
            optimizer=optimizer,
            loss=keras.losses.Huber(),
        )

        self.target_q_network.set_weights(self.q_network.get_weights())


        # print("q in/out:", self.q_network.input_shape, "->", self.q_network.output_shape)
        # print("t in/out:", self.target_q_network.input_shape, "->", self.target_q_network.output_shape)


        # self.q0 = self.q_network.predict(self.states[None, :])
        # self.t0 = self.target_q_network.predict(self.states[None, :])
        # print("same init preds?", np.allclose(self.q0, self.t0))
        
        # print(self.states[None, :])
        # print(self.states)
        # print(f"q0: {self.q0}")
        # print(f"t0: {self.t0}")

        

        
    def ft_choose_action(self, state_vec):
        q_values = self.q_network.predict(state_vec[None, :], verbose=0)[0]

        u = random.uniform(0, 1)
        if u < self.epsilon:
            action = random.randrange(self.num_actions)
        else:
            action = int(np.argmax(q_values))

        self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)  

        # print(f'q: {q_values}')
        # print(f"u: {u}")
        # print(f"action taken: {self.ACTIONS[action]}")
        # print(f"epsilon: {self.epsilon}")
        return action
        # q_values = self.q_network.predict(state_vec[None, :], verbose=0)[0]

        # # mask l'action opposée si serpent >=2
        # mask = np.zeros_like(q_values, dtype=bool)
        # if len(self.game.snake) >= 2 and self.last_action is not None:
        #     opp = {0:1, 1:0, 2:3, 3:2}  # up<->down, left<->right
        #     mask[opp[self.last_action]] = True
        #     q_values = np.where(mask, -1e9, q_values)

        # if random.random() < self.epsilon:
        #     # explore mais respecte le masque
        #     valid = [i for i in range(self.num_actions) if not mask[i]]
        #     action = random.choice(valid)
        # else:
        #     action = int(np.argmax(q_values))

        # self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        # self.last_action = action
        # return action


    # Replay buffer
    def ft_store_experience(self, state, action_index, reward, next_state, is_over):
        experience = state, action_index, reward, next_state, is_over
        
        self.experience.append(experience)



    def ft_calculate_reward(self, code, is_over):
        if is_over:
            return -1.0
        elif code == self.GREEN_APPLE:
            return +0.1
        elif code == self.RED_APPLE:
            return -0.1
        elif code == self.EMPTY:
            return -0.001
        else:
            return 0.0  

        # R_DEATH = -5.0
        # R_GREEN = +5.0
        # R_RED   = -2.0
        # R_STEP  = -0.01

        # if is_over:
        #     return R_DEATH
        # if code == self.GREEN_APPLE:
        #     return R_GREEN
        # if code == self.RED_APPLE:
        #     return R_RED

        # # shaping: se rapprocher de la verte la plus proche
        # # (différence de distance de Manhattan head -> verte la plus proche)
        # head = tuple(self.game.snake[0])
        # if len(self.game.green_apples) > 0:
        #     d_before = min(abs(head[0]-r)+abs(head[1]-c) for r,c in self.game.green_apples)
        # else:
        #     d_before = 0


        # head2 = tuple(self.game.snake[0])
        # if len(self.game.green_apples) > 0:
        #     d_after = min(abs(head2[0]-r)+abs(head2[1]-c) for r,c in self.game.green_apples)
        # else:
        #     d_after = d_before

        # shaping = 0.05 * (d_before - d_after)  # >0 si on se rapproche
        # return R_STEP + shaping



    def ft_train_step(self):

        if len(self.experience) < max(self.minibatch_size, self.warmup_steps):
            return

        batch = random.sample(self.experience, self.minibatch_size)
    
        states      = np.array([exp[0] for exp in batch], dtype=np.float32)
        actions     = np.array([exp[1] for exp in batch], dtype=np.int32)
        rewards     = np.array([exp[2] for exp in batch], dtype=np.float32)
        next_states = np.array([exp[3] for exp in batch], dtype=np.float32)
        dones       = np.array([float(exp[4]) for exp in batch], dtype=np.float32)


        q_next_online = self.q_network.predict(next_states, verbose=0)
        best_next_act = np.argmax(q_next_online, axis=1)
        q_next_target = self.target_q_network.predict(next_states, verbose=0)
        target_next = q_next_target[np.arange(len(batch)), best_next_act]
        y = rewards + self.gamma * (1 - dones) * target_next



        q_pred = self.q_network.predict(states, verbose=0)

        B = len(batch)
        for i in range(B):
            q_pred[i, actions[i]] = y[i]

        loss = self.q_network.train_on_batch(states, q_pred)

        # self.step_count += 1
        # if self.step_count % 2 != 0:
        #     self.target_q_network.set_weights(self.q_network.get_weights())
            

        self.step_count += 1
        self._soft_update(tau=0.005)

        return float(loss)

    def _soft_update(self, tau=0.005):
        qW = self.q_network.get_weights()
        tW = self.target_q_network.get_weights()
        # ⚠️ formule correcte :
        newW = [tau * qw + (1 - tau) * tw for qw, tw in zip(qW, tW)]
        self.target_q_network.set_weights(newW)




    def ft_interaction(self, sessions: int):
        # --- init DQN ---
        self.experience = deque(maxlen=self.memory_size)
        self.states = self.ft_encode_vision(self.game.ft_get_vision_snake())
        self.state_size = (self.states.shape[0],)
        self.num_actions = len(self.ACTIONS)
        self.ft_define_initial_model()
        self.step_count = 0

        # --- stats globales ---
        max_length_global = 0
        max_duration_global = 0

        for session_id in range(1, sessions + 1):
            # --- reset session ---
            self.is_over = False
            self.last_action = None
            self.states = self.ft_encode_vision(self.game.ft_get_vision_snake())

            start_len = len(self.game.snake)
            start_head = tuple(self.game.snake[0])
            duration = 0
            ep_return = 0.0
            last_loss = None
            last_event_tile = "EMPTY"


            print(f"\n=== SESSION {session_id} START ===")
            print(f"→ start_len: {start_len}, start_head: {start_head}, epsilon: {self.epsilon:.3f}")

            try:
                while not self.is_over:
                    action_index = self.ft_choose_action(self.states)
                    action_str = self.ACTIONS[action_index]

                    code, is_over = self.game.ft_move_snake(action_str, "off")

                    reward = self.ft_calculate_reward(code, is_over)
                    ep_return += reward

                    next_state = self.ft_encode_vision(self.game.ft_get_vision_snake())
                    self.ft_store_experience(self.states, action_index, reward, next_state, is_over)
                    loss = self.ft_train_step()
                    if loss is not None:
                        last_loss = loss

                    self.states = next_state
                    self.is_over = is_over
                    duration += 1
                    len_after = len(self.game.snake)

                    if len_after > max_length_global:
                        max_length_global = len_after

                    last_event_tile = self.CHARS[code]

                    # print(
                    #     f"step={duration:03d} | act={action_str:<5} | tile={last_event_tile:<10} "
                    #     f"| len={len_after:<2d} | reward={reward:+.3f} | eps={self.epsilon:.3f} | is_over={is_over}"
                    # )

            except Exception as e:
                print("⚠️  Exception pendant la session !")
                print(traceback.format_exc())
                self.is_over = True

            # --- fin session ---
            if duration > max_duration_global:
                max_duration_global = duration

            print("\n--- SESSION SUMMARY ---")
            print(f"End len: {len(self.game.snake)}")
            print(f"Dead: {self.is_over}")
            print(f"Duration: {duration}")
            print(f"Episode return: {ep_return:.4f}")
            print(f"Final epsilon: {self.epsilon:.4f}")
            print(f"Last loss: {last_loss if last_loss is not None else 'None'}")
            print(f"Last event: {last_event_tile}")
            print(f"Global max len: {max_length_global}")
            print(f"Global max duration: {max_duration_global}")

            # --- reset jeu ---
            self.game.ft_reset()
            print(f"After reset: len={len(self.game.snake)}")
            print(f"=== SESSION {session_id} END ===\n")

        return max_length_global, max_duration_global



    def __str__(self):
        
        return 'TEST'





def main():

    size = 10
    game = Game(size, size)
    networks = DeepQNetwork(game)

    sessions = 50

    max_length, max_duration = networks.ft_interaction(sessions)

    print(f"Game over, max length: {max_length}, max_duration: {max_duration}")
    # print(f"Weights: \n{networks.q_network.get_weights()}")

    [...]


if __name__ == "__main__":
    main()