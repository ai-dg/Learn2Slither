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
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory_size = 100_000
        self.num_steps_for_update = 200

        self.q_network = None
        self.target_q_network = None

        
        self.minibatch_size = 64
        self.game = game

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
            loss=MSE(),
        )
        self.target_q_network.compile(
            optimizer=optimizer,
            loss=MSE(),
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


    def ft_train_step(self):

        if len(self.experience) < self.minibatch_size:
            return

        batch = random.sample(self.experience, self.minibatch_size)
    
        states      = np.array([exp[0] for exp in batch], dtype=np.float32)
        actions     = np.array([exp[1] for exp in batch], dtype=np.int32)
        rewards     = np.array([exp[2] for exp in batch], dtype=np.float32)
        next_states = np.array([exp[3] for exp in batch], dtype=np.float32)
        dones       = np.array([float(exp[4]) for exp in batch], dtype=np.float32)


        q_next = self.target_q_network.predict(next_states, verbose=0)
        max_next = np.max(q_next, axis=1)

        y = rewards + self.gamma * (1 - dones) * max_next


        q_pred = self.q_network.predict(states, verbose=0)

        B = len(batch)
        for i in range(B):
            q_pred[i, actions[i]] = y[i]

        loss = self.q_network.train_on_batch(states, q_pred)

        self.step_count += 1
        if self.step_count % self.num_steps_for_update == 0:
            self.target_q_network.set_weights(self.q_network.get_weights())

        return float(loss)



    def ft_interaction(self, sessions):

        self.experience = deque(maxlen=self.memory_size)
        self.states = self.ft_encode_vision(self.game.ft_get_vision_snake())
        self.state_size = (self.states.shape[0],)
        self.num_actions = len(self.ACTIONS)
        self.ft_define_initial_model()
        self.step_count = 0

        max_length = 0
        max_duration = 0
        current_session = 0
        for nbr in range(sessions):

            self.is_over = False
            self.states = self.ft_encode_vision(self.game.ft_get_vision_snake())
            # self.state_size = (self.states.shape[0],)
            duration = 0
            current_session += 1
            print(f"current_session: {current_session}")
            loss = 0
            ep_return = 0.0
            len_before = 0
            name = ''
            while not self.is_over:
                self.action_index = self.ft_choose_action(self.states)
                self.action_str = self.ACTIONS[self.action_index]
                len_before = len(self.game.snake)
                self.code, self.is_over = self.game.ft_move_snake(self.action_str, 'off')
            
                self.reward = self.ft_calculate_reward(self.code, self.is_over)
                ep_return += self.reward
                self.next_state = self.ft_encode_vision(self.game.ft_get_vision_snake())
                self.ft_store_experience(self.states, self.action_index, self.reward, self.next_state, self.is_over)

                # print(self.experience)

                loss = self.ft_train_step()

                self.states = self.next_state
                duration += 1
                if len(self.game.snake) > max_length:
                    max_length = len(self.game.snake)

                name  = self.CHARS[self.code]
                
            
            print(f"loss: {loss}")
            print(f"ep_return: {ep_return}")
            print(f"epsilon: {self.epsilon}")
            print(f"[eat] code={self.code} len_before={len_before} len_after={len(self.game.snake)}")
            print(f"[event] tile={name} len={len(self.game.snake)} eps={self.epsilon:.3f}")

            
            if duration > max_duration:
                max_duration = duration
            self.game.ft_reset()

            
        
        return max_length, max_duration
        



    def __str__(self):
        
        return 'TEST'





def main():

    size = 10
    game = Game(size, size)
    networks = DeepQNetwork(game)

    sessions = 100

    max_length, max_duration = networks.ft_interaction(sessions)

    print(f"Game over, max length: {max_length}, max_duration: {max_duration}")
    # print(f"Weights: \n{networks.q_network.get_weights()}")

    [...]


if __name__ == "__main__":
    main()