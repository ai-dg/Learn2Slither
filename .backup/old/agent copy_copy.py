from os import write
import numpy as np
import tensorflow as tf
from game_data import Game
from collections import deque
import random
from tabulate import tabulate
import traceback
# import keras
# from keras.layers import Dense, Input
# from keras.models import Sequential
# from keras.optimizers import Adam
import sys
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

tf.config.set_visible_devices([], 'GPU')

TAU = 1e-3
NUM_STEPS_FOR_UPDATE = 1
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

    # debug switches
    debug_train = True        # logs d’entraînement
    log_every = 50            # fréquence de logs (train-steps)


    def ft_encode_vision(self, vision):
        parts_state = []
        for a in self.ACTIONS:
            parts_state.extend(vision[a])
        states = np.asarray(parts_state, np.float32) / 5
        return states

    def __init__(self, game: Game):
        # Hyperparams
        self.gamma = 0.995
        self.alpha = 1e-3
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.memory_size = 100_000
        self.minibatch_size = MINIBATCH_SIZE
        self.warmup_steps = 500

        # Models & optimizer
        self.q_network = None
        self.target_q_network = None
        self.optimizer = None
        self.loss_fn = keras.losses.MeanSquaredError()

        # Environment
        self.game = game
        self.experience = deque(maxlen=self.memory_size)

        # States
        self.states = None
        self.state_size = None
        self.num_actions = len(self.ACTIONS)

        # Runtime variables
        self.train_step_count = 0
        self.duration = 0
        self.step_count = 0
        self.is_over = False
        self.last_action = None

        # Global stats
        self.max_length_global = 0
        self.max_duration_global = 0



    # agent.py  (inside ft_define_initial_model)

    def ft_define_initial_model(self):
        self.q_network = Sequential([
            Input(shape=self.state_size),
            Dense(64, activation='relu', name='layer1'),
            Dense(64, activation='relu', name='layer2'),
            Dense(self.num_actions, activation='linear'),
        ])
        self.target_q_network = Sequential([
            Input(shape=self.state_size),
            Dense(64, activation='relu', name='layer1'),
            Dense(64, activation='relu', name='layer2'),
            Dense(self.num_actions, activation='linear'),
        ])

        self.optimizer = Adam(learning_rate=self.alpha)
        self.q_network.compile(optimizer=self.optimizer, loss=self.loss_fn)
        self.target_q_network.compile(optimizer=self.optimizer, loss=self.loss_fn)

        # ⬇️ Appel symbolique pour définir .input / .output (compatible avec le test)
        sym = keras.Input(shape=self.state_size, name="build_input")
        _ = self.q_network(sym)
        _ = self.target_q_network(sym)

        # try:
        #     self.q_network.load_weights('./models/600sess.weights.h5')
        #     print("[info] Loaded weights from ./model.weights.h5")
        # except Exception:
        #     print("[info] No existing weights found, starting fresh.")


        # Sync initiale
        self.target_q_network.set_weights(self.q_network.get_weights())



    # def ft_choose_action(self, state_vec):
    #     q_values = self.q_network.predict(state_vec[None, :], verbose=0)[0]
    #     u = random.uniform(0, 1)
    #     if u < self.epsilon:
    #         action = random.randrange(self.num_actions)   # exploration
    #     else:
    #         action = int(np.argmax(q_values))             # exploitation
    #     return action

    def ft_choose_action(self, state_vec):
        if len(self.experience) < self.warmup_steps:
            return random.randrange(self.num_actions)  # exploration pure pendant warmup
        q_values = self.q_network.predict(state_vec[None, :], verbose=0)[0]
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        return int(np.argmax(q_values))

    # Replay buffer
    def ft_store_experience(self, state, action_index, reward, next_state, is_over):
        self.experience.append((
            np.array(state, dtype=np.float32, copy=True),
            int(action_index),
            float(reward),
            np.array(next_state, dtype=np.float32, copy=True),
            bool(is_over),
        ))

    def ft_calculate_reward(self, code, is_over, next_state):
        # next_state est normalisé (/5). Reviens aux codes entiers:
        vision_tiles = (next_state * 5).astype(int)

        # Terminal / événements
        if is_over:
            return -12.0
        if code == self.GREEN_APPLE:
            return +10.0
        if code == self.RED_APPLE:
            return -4.0

        # Shaping doux basé vision (faible amplitude)
        score = 0.0
        if self.GREEN_APPLE in vision_tiles:
            score += 0.03
        if self.RED_APPLE in vision_tiles:
            score -= 0.02

        # Petite pénalité temps pour éviter de tourner en rond
        score -= 0.0015
        return score

    def _check_nan(self, name, arr):
        if np.isnan(arr).any():
            print(f"[WARN] NaN detected in {name}")

    def ft_compute_loss(self):
        """Calcule la loss du mini-batch (targets gelées, q_values sélectionnées)."""
        batch = random.sample(self.experience, k=MINIBATCH_SIZE)

        states      = np.array([exp[0] for exp in batch], dtype=np.float32)
        actions     = np.array([exp[1] for exp in batch], dtype=np.int32)
        rewards     = np.array([exp[2] for exp in batch], dtype=np.float32)
        next_states = np.array([exp[3] for exp in batch], dtype=np.float32)
        dones       = np.array([float(exp[4]) for exp in batch], dtype=np.float32)

        # Q_target(next_state, a') -> max_a'
        q_next_target = self.target_q_network(next_states, training=False)  # (B, A)
        max_qsa = tf.reduce_max(q_next_target, axis=1)                      # (B,)

        # y = r + gamma * (1 - done) * max Q_target
        y_targets = rewards + (self.gamma * max_qsa * (1.0 - dones))
        y_targets = tf.stop_gradient(y_targets)  # ✅ ne pas backprop à travers les targets

        # Q_policy(state, a) (sélectionner l’action jouée)
        q_values_all = self.q_network(states, training=True)                # (B, A)
        idx = tf.stack([tf.range(tf.shape(q_values_all)[0]),
                        tf.cast(actions, tf.int32)], axis=1)               # (B, 2)
        q_values = tf.gather_nd(q_values_all, idx)                          # (B,)

        # Sanity logs
        if self.debug_train and (self.train_step_count % self.log_every == 0):
            print(f"[train:{self.train_step_count}] "
                  f"y_targets mean={float(tf.reduce_mean(y_targets)):.6f} | "
                  f"q_values mean={float(tf.reduce_mean(q_values)):.6f}")

        # ✅ Loss correctement appliquée
        loss = self.loss_fn(y_targets, q_values)                            # scalaire

        return loss, q_values, y_targets

    def ft_update_target_soft_update(self):
        """Soft update: θ_target ← τ θ_policy + (1-τ) θ_target"""
        for target_weights, q_net_weights in zip(
            self.target_q_network.weights, self.q_network.weights
        ):
            target_weights.assign(TAU * q_net_weights + (1.0 - TAU) * target_weights)

    def ft_agent_learn(self):
        """Un train-step (mini-batch) avec GradientTape + logs."""
        if len(self.experience) < max(MINIBATCH_SIZE, self.warmup_steps):
            if self.debug_train and (len(self.experience) % 25 == 0):
                print(f"[train] warmup: mem={len(self.experience)} < {max(MINIBATCH_SIZE, self.warmup_steps)}")
            return None
        
        

        self.train_step_count += 1

        with tf.GradientTape() as tape:
            loss, q_values, y_targets = self.ft_compute_loss()

        gradients = tape.gradient(loss, self.q_network.trainable_variables)

        # Guard: gradients None ?
        if any(g is None for g in gradients):
            print(f"[WARN] None gradients at step {self.train_step_count}")
        else:
            grad_norm = tf.linalg.global_norm([g for g in gradients if g is not None])
            if self.debug_train and (self.train_step_count % self.log_every == 0):
                print(f"[train:{self.train_step_count}] loss={float(loss):.6f} | grad_norm={float(grad_norm):.6f}")

        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))

        # Soft update du target
        self.ft_update_target_soft_update()

        self.updates_applied += 1  # via nonlocal ou self.updates_applied
        # log rolling
        self.loss_hist.append(float(loss))
        self.td = (y_targets - q_values)  # tf.Tensor
        self.td_mean_hist.append(float(tf.reduce_mean(self.td)))
        self.td_std_hist.append(float(tf.math.reduce_std(self.td)))

        return float(loss)

    def ft_check_update_conditions(self, step):
        return (step + 1) % NUM_STEPS_FOR_UPDATE == 0 and len(self.experience) >= MINIBATCH_SIZE


    # def _budget_for_target_length(self, total_sessions: int):
    #     """
    #     Calcule un nombre total de pas maximum (max_num_timesteps)
    #     pour chaque session, en fonction du nombre total de sessions.

    #     - Peu de sessions => sessions plus courtes
    #     - Beaucoup de sessions => sessions plus longues
    #     """
    #     base = 10
    #     scale = 2.5
    #     return int(base + scale * total_sessions)

    def _budget_for_target_length(self, total_sessions: int):
        # Croissant, plafonné pour éviter les épisodes interminables
        base = 150       # steps mini
        slope = 0.9      # +0.9 step par session
        cap = 1200       # plafond par session
        return int(min(base + slope * total_sessions, cap))



    def ft_interaction(self, sessions: int):
        from collections import deque
        W = 100  # fenêtre mobile
        self.len_hist = deque(maxlen=W)
        self.dur_hist = deque(maxlen=W)
        self.ret_hist = deque(maxlen=W)
        self.death_causes = deque(maxlen=W)  # 'WALL' / 'BODY' / 'LEN0' / etc.

        self.updates_applied = 0
        self.loss_hist = deque(maxlen=W)
        self.td_mean_hist = deque(maxlen=W)
        self.td_std_hist = deque(maxlen=W)

        action_counts = {a: 0 for a in self.ACTIONS}


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
        
        # max_num_timesteps = self._budget_for_target_length(sessions)

        max_num_timesteps = 1000
        green_hits = 0
        red_hits = 0

        for session_id in range(1, sessions + 1):
            # --- reset session ---
            self.is_over = False
            self.last_action = None
            self.states = self.ft_encode_vision(self.game.ft_get_vision_snake())

            start_len = len(self.game.snake)
            start_head = tuple(self.game.snake[0])
            ep_return = 0.0
            last_loss = None
            last_event_tile = "EMPTY"

            # print(f"\n=== SESSION {session_id} START ===")
            # print(f"→ start_len: {start_len}, start_head: {start_head}, epsilon: {self.epsilon:.3f}")

            # step = 0
            
            
            # print(f"max_num_timesteps: {max_num_timesteps}")

            self.duration = 0
            try:
                # while not self.is_over:
                for step in range(max_num_timesteps):
                    self.duration += 1
                    # step += 1
                    action_index = self.ft_choose_action(self.states)

                    action_counts[self.ACTIONS[action_index]] += 1


                    action_str = self.ACTIONS[action_index]

                    code, is_over = self.game.ft_move_snake(action_str, "off")

                    next_state = self.ft_encode_vision(self.game.ft_get_vision_snake())
                    
                    reward = self.ft_calculate_reward(code, is_over, next_state)
                    # print(f"Code {code}")
                    # print(f"Code {self.CHARS[code]}")
                    if code == self.GREEN_APPLE: green_hits += 1
                    if code == self.RED_APPLE:   red_hits += 1
                    ep_return += reward


                    # Store after getting next_state
                    self.ft_store_experience(self.states, action_index, reward, next_state, is_over)

                    # Learn periodically
                    if self.ft_check_update_conditions(step):
                        loss = self.ft_agent_learn()
                    else:
                        loss = None
                    if loss is not None:
                        last_loss = loss



                    self.states = next_state.copy()
                    self.is_over = is_over

                    

                    len_after = len(self.game.snake)

                    if len_after > max_length_global:
                        max_length_global = len_after

                    last_event_tile = self.CHARS[code]

                    if is_over:
                        break

            except Exception as e:
                print("⚠️  Exception pendant la session !")
                print(traceback.format_exc())
                self.is_over = True
                sys.exit(1)

            # --- fin session ---
            if self.duration > max_duration_global:
                max_duration_global = self.duration

            # Décroissance d’epsilon **par épisode**
            old_eps = self.epsilon
            # self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            # après le reset de fin d'épisode
            if session_id < 200:
                self.epsilon = max(0.2, self.epsilon * self.epsilon_decay)  # reste ≥0.2
            else:
                self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


            # print("\n--- SESSION SUMMARY ---")
            # print(f"End len: {len(self.game.snake)}")
            # print(f"Dead: {self.is_over}")
            # print(f"Duration: {self.duration}")
            # print(f"Episode return: {ep_return:.4f}")
            # print(f"Epsilon: {old_eps:.4f} -> {self.epsilon:.4f}")
            # print(f"Last loss: {last_loss if last_loss is not None else 'None'}")
            # print(f"Last event: {last_event_tile}")
            # print(f"Global max len: {max_length_global}")
            # print(f"Global max learning duration: {max_duration_global}")

            self.len_hist.append(len(self.game.snake))
            self.dur_hist.append(self.duration)
            self.ret_hist.append(ep_return)
            self.death_causes.append(last_event_tile)  # 'WALL' / 'BODY' / etc.

            # résumé compact toutes les 100 sessions
            if session_id % 100 == 0:
                print(f"[REWARDS/100] sess={session_id} green={green_hits} red={red_hits}")
                green_hits = 0; red_hits = 0
            if session_id % 100 == 0:
                def avg(x): return float(np.mean(x)) if x else float('nan')
                def med(x): return float(np.median(x)) if x else float('nan')
                def p(x, q): return float(np.percentile(x, q)) if x else float('nan')
                def rate(th): return 100.0 * sum(1 for v in self.len_hist if v >= th) / len(self.len_hist) if self.len_hist else 0.0
                from collections import Counter
                deaths = Counter(self.death_causes)
                actions = ", ".join(f"{k}:{v}" for k,v in action_counts.items())
                
                print(f"\n[ROLLING/{W}] sess={session_id} "
                    f"| len (avg/med/max p90) = {avg(self.len_hist):.2f}/{med(self.len_hist):.2f}/{(max(self.len_hist) if self.len_hist else 0):.0f}/{p(self.len_hist,90):.0f} "
                    f"| dur avg={avg(self.dur_hist):.1f} "
                    f"| ret avg={avg(self.ret_hist):.2f} "
                    f"| ε={self.epsilon:.3f} "
                    f"| buf={len(self.experience)} "
                    f"| updates={self.updates_applied} "
                    f"| loss avg={avg(self.loss_hist):.4f} (±{np.std(self.loss_hist):.4f}) "
                    f"| TD μ={avg(self.td_mean_hist):.4f} σ={avg(self.td_std_hist):.4f} "
                    f"| ≥10:{rate(10):.1f}% ≥15:{rate(15):.1f}% ≥20:{rate(20):.1f}% ≥25:{rate(25):.1f}% ≥30:{rate(30):.1f}% "
                    f"| deaths={dict(deaths)} "
                    f"| actions={actions}")
                # reset visibles-only counters
                action_counts = {a: 0 for a in self.ACTIONS}
            if self.train_step_count % 1000 == 0:
                print(f"[INFO] total_experience={len(self.experience)}")

            if session_id % 500 == 0:
                print(f"=== SESSION {session_id} END ===\n")
                print(f"Global max len: {max_length_global}")
                self.q_network.save_weights(f'./models/{session_id}sess.weights.h5')

            # --- reset jeu ---
            self.game.ft_reset()
            # print(f"After reset: len={len(self.game.snake)}")
            # print(f"=== SESSION {session_id} END ===\n")
            

        return max_length_global, max_duration_global

    def ft_reset(self):
        self.__init__(self.game)
        

    def __str__(self):
        return 'TEST'
    
    # type hints (indicatifs)
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
    duration: int
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
   
        max_lengths = []

        sessions = 500
        max_length, max_duration = networks.ft_interaction(sessions)
        max_lengths.append(max_length)
        print(f"Game over, max length: {max_length}, max_duration: {max_duration}")
        # networks.ft_reset()
        
        networks.q_network.save_weights('./standard.weights.h5')
    except KeyboardInterrupt:
        del game
        del networks
        sys.exit(1)

if __name__ == "__main__":
    main()
