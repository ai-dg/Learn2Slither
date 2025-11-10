import sys
import random
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from game_data import Game

# tf.config.set_visible_devices([], 'GPU')  # GPU activé après mise à jour CuDNN

# Vérifier la disponibilité du GPU
print("=== CONFIGURATION GPU ===")
print(f"GPU disponible: {tf.config.list_physical_devices('GPU')}")
print(f"GPU détecté: {len(tf.config.list_physical_devices('GPU')) > 0}")
if tf.config.list_physical_devices('GPU'):
    print("✅ GPU sera utilisé pour l'entraînement")
else:
    print("❌ Pas de GPU détecté, utilisation du CPU")
print("=========================")

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
    CHARS = ['EMPTY', 'WALL', 'HEAD', 'BODY', 'RED_APPLE', 'GREEN_APPLE']

    def ft_encode_vision(self, vision):
        parts_state = []
        for a in self.ACTIONS:
            if a in vision and vision[a]:
                # Prendre les 3 premiers éléments de chaque direction
                for i in range(3):
                    if i < len(vision[a]):
                        parts_state.append(vision[a][i] / 5.0)  # Normaliser par 5
                    else:
                        parts_state.append(0.0)  # EMPTY si pas d'élément
            else:
                # Si pas de vision, 3 zéros
                parts_state.extend([0.0, 0.0, 0.0])
        return np.asarray(parts_state, np.float32)

    def __init__(self, game: Game):
        self.gamma = 0.95  # Facteur de discount plus élevé pour la planification à long terme
        self.alpha = 5e-4  # Taux d'apprentissage plus conservateur
        self.epsilon = 1.0
        self.epsilon_min = 0.1  # Taux d'exploration minimum plus élevé pour plus d'exploration
        self.epsilon_decay = 0.995  # Décroissance plus rapide de l'exploration
        self.epsilon_exploit = 0.05  # Taux d'exploration en exploitation plus élevé
        self.memory_size = 100_000  # Taille du buffer de replay plus grande
        self.minibatch_size = 64  # Taille du minibatch plus grande
        self.warmup_steps = 100  # Plus d'étapes avant d'utiliser le réseau
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
        self.loss = 0.0

    def ft_define_initial_model(self):
        # Architecture plus profonde et plus large pour une meilleure capacité d'apprentissage
        self.q_network = Sequential([
            Input(shape=self.state_size),
            Dense(256, activation='relu'),
            Dropout(0.2),  # Dropout pour éviter l'overfitting
            Dense(256, activation='relu'),
            Dropout(0.2),
            Dense(128, activation='relu'),
            Dropout(0.1),
            Dense(64, activation='relu'),
            Dense(self.num_actions, activation='linear'),
        ])
        self.target_q_network = Sequential([
            Input(shape=self.state_size),
            Dense(256, activation='relu'),
            Dropout(0.2),
            Dense(256, activation='relu'),
            Dropout(0.2),
            Dense(128, activation='relu'),
            Dropout(0.1),
            Dense(64, activation='relu'),
            Dense(self.num_actions, activation='linear'),
        ])
        # Optimiseur avec learning rate adaptatif
        self.optimizer = Adam(learning_rate=self.alpha, beta_1=0.9, beta_2=0.999)
        self.q_network.compile(optimizer=self.optimizer, loss=self.loss_fn)
        self.target_q_network.compile(optimizer=self.optimizer, loss=self.loss_fn)
        sym = keras.Input(shape=self.state_size)
        _ = self.q_network(sym)
        _ = self.target_q_network(sym)
        self.target_q_network.set_weights(self.q_network.get_weights())

    def ft_choose_action(self, state_vec, epsilon_curr: float):
        if len(self.experience) < self.warmup_steps:
            # Pendant le warmup, exploration pure avec évitement des murs
            return self.ft_choose_safe_action(state_vec, avoid_walls_only=True)
        
        q_values = self.q_network.predict(state_vec[None, :], verbose=0)[0]
        
        # Séparation claire entre exploration et exploitation
        if random.random() < epsilon_curr:
            # EXPLORATION : Prendre des risques calculés pour découvrir de nouvelles stratégies
            return self.ft_choose_exploration_action(state_vec, q_values)
        else:
            # EXPLOITATION : Utiliser les connaissances apprises
            return int(np.argmax(q_values))
    
    def ft_choose_exploration_action(self, state_vec, q_values):
        """Action d'exploration qui privilégie fortement les pommes vertes"""
        # Chercher les pommes vertes en priorité
        green_apple_actions = []
        safe_actions = []
        
        for i, action in enumerate(self.ACTIONS):
            direction_start = i * 3
            if direction_start + 2 < len(state_vec):
                first_element = state_vec[direction_start] * 5
                
                if first_element == 4:  # Pomme verte visible
                    green_apple_actions.append(i)
                elif first_element not in [1, 3]:  # Pas un mur ni le corps
                    safe_actions.append(i)
        
        # Si on voit une pomme verte, la prendre avec 80% de probabilité
        if green_apple_actions:
            if random.random() < 0.8:
                return random.choice(green_apple_actions)
            else:
                # 20% du temps, choisir parmi toutes les actions
                return random.choice(list(range(len(self.ACTIONS))))
        
        # Sinon, choisir parmi les actions sûres
        if safe_actions:
            return random.choice(safe_actions)
        
        # Dernière option : choisir aléatoirement
        return random.randrange(len(self.ACTIONS))
    
    def ft_choose_safe_action(self, state_vec, avoid_walls_only=False):
        """Choisir une action qui évite les murs (et optionnellement le corps)"""
        # state_vec contient 12 valeurs : 3 par direction (up, down, left, right)
        # Les 3 premières valeurs sont pour 'up', les 3 suivantes pour 'down', etc.
        
        safe_actions = []
        for i, action in enumerate(self.ACTIONS):
            # Vérifier si la direction mène à un mur (code 1)
            direction_start = i * 3
            if direction_start + 2 < len(state_vec):
                # Le premier élément de chaque direction (normalisé par 5)
                first_element = state_vec[direction_start] * 5  # Dénormaliser
                
                if avoid_walls_only:
                    # Éviter seulement les murs
                    if first_element != 1:  # Pas un mur
                        safe_actions.append(i)
                else:
                    # Éviter murs et corps
                    if first_element not in [1, 3]:  # Pas un mur ni le corps
                        safe_actions.append(i)
        
        if safe_actions:
            return random.choice(safe_actions)
        else:
            # Si toutes les directions sont dangereuses, choisir celle qui mène au corps plutôt qu'au mur
            least_dangerous = []
            for i, action in enumerate(self.ACTIONS):
                direction_start = i * 3
                if direction_start + 2 < len(state_vec):
                    first_element = state_vec[direction_start] * 5
                    if first_element == 3:  # Corps (moins dangereux qu'un mur)
                        least_dangerous.append(i)
            
            if least_dangerous:
                return random.choice(least_dangerous)
            else:
                # Dernière option : choisir aléatoirement
                return random.randrange(self.num_actions)

    def ft_store_experience(self, state, action_index, reward, next_state, is_over):
        self.experience.append((
            np.array(state, dtype=np.float32, copy=True),
            int(action_index),
            float(reward),
            np.array(next_state, dtype=np.float32, copy=True),
            bool(is_over),
        ))

    def ft_calculate_reward(self, code, is_over, next_state):
        reward = 0.0
        
        if is_over:
            reward = -100.0  # Pénalité forte pour la mort
        elif code == self.GREEN_APPLE:
            reward = 50.0  # Récompense encore plus forte pour pomme verte
        elif code == self.RED_APPLE:
            reward = -25.0  # Pénalité forte pour pomme rouge
        
        # Récompense pour voir une pomme verte dans la vision - plus agressive
        green_apples_seen = 0
        red_apples_seen = 0
        
        for i in range(0, len(next_state), 3):
            if i + 2 < len(next_state):
                first_element = next_state[i] * 5  # Dénormaliser
                if first_element == 4:  # Pomme verte visible
                    green_apples_seen += 1
                elif first_element == 5:  # Pomme rouge visible
                    red_apples_seen += 1
        
        # Récompenses basées sur ce qui est visible - plus agressives
        reward += green_apples_seen * 5.0  # Récompense forte pour chaque pomme verte visible
        reward -= red_apples_seen * 2.0   # Pénalité pour chaque pomme rouge visible
        
        # Récompense de survie progressive
        survival_reward = 1.0
        reward += survival_reward
        
        # Récompense pour s'approcher des pommes vertes
        for i in range(0, len(next_state), 3):
            if i + 2 < len(next_state):
                first_element = next_state[i] * 5
                if first_element == 4:  # Pomme verte visible
                    # Récompense inversement proportionnelle à la distance
                    distance = 3  # Approximation de la distance
                    reward += 3.0 / distance  # Plus proche = plus de récompense
        
        # Récompense pour éviter les pommes rouges
        for i in range(0, len(next_state), 3):
            if i + 2 < len(next_state):
                first_element = next_state[i] * 5
                if first_element == 5:  # Pomme rouge visible
                    # Pénalité inversement proportionnelle à la distance
                    distance = 3  # Approximation de la distance
                    reward -= 1.5 / distance  # Plus proche = plus de pénalité
        
        return reward

    def ft_compute_loss(self):
        batch = random.sample(self.experience, k=self.minibatch_size)
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
        idx = tf.stack([tf.range(tf.shape(q_values_all)[0]), tf.cast(actions, tf.int32)], axis=1)
        q_values = tf.gather_nd(q_values_all, idx)
        loss = self.loss_fn(y_targets, q_values)
        return loss

    def ft_update_target_soft_update(self):
        for tw, qw in zip(self.target_q_network.weights, self.q_network.weights):
            tw.assign(TAU * qw + (1.0 - TAU) * tw)

    def ft_agent_learn(self):
        if len(self.experience) < max(self.minibatch_size, self.warmup_steps):
            return None
        self.train_step_count += 1
        with tf.GradientTape() as tape:
            loss = self.ft_compute_loss()
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        
        # Logs détaillés pour diagnostic
        # if self.train_step_count % 100 == 0:
        #     grad_norms = [tf.norm(g).numpy() for g in gradients if g is not None] if gradients else [0]
        #     sample_states = np.array([exp[0] for exp in list(self.experience)[-10:]])
        #     q_values = self.q_network.predict(sample_states, verbose=0)
            
            # print(f"[LEARN] Step {self.train_step_count}:")
            # print(f"  Loss: {float(loss):.4f}")
            # print(f"  Grad norm: {np.mean(grad_norms):.4f} (max: {np.max(grad_norms):.4f})")
            # print(f"  Q-values: mean={np.mean(q_values):.4f}, std={np.std(q_values):.4f}")
            # print(f"  Epsilon: {self.epsilon:.4f}")
            # print(f"  Buffer size: {len(self.experience)}")
        
        if gradients is not None:
            self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
        self.ft_update_target_soft_update()
        
        if self.train_step_count % 1000 == 0:
            print(f"train step: {self.train_step_count} loss: {loss} len_exp:{len(self.experience)}")
        return float(loss)

    def ft_check_update_conditions(self, step):
        return (step + 1) % NUM_STEPS_FOR_UPDATE == 0 and len(self.experience) >= self.minibatch_size

    def ft_interaction(self, sessions: int):
        green_hits_win = 0
        red_hits_win = 0
        self.experience = deque(maxlen=self.memory_size)
        self.len_hist = deque(maxlen=100)
        self.states = self.ft_encode_vision(self.game.ft_get_vision_snake())
        self.state_size = (self.states.shape[0],)
        self.num_actions = len(self.ACTIONS)
        self.ft_define_initial_model()
        self.step_count = 0
        max_length_global = 0
        max_duration_global = 0
        max_num_timesteps = 1000
        epsilon_curr = 0

        for session_id in range(1, sessions + 1):
            self.is_over = False
            self.last_action = None
            self.states = self.ft_encode_vision(self.game.ft_get_vision_snake())
            self.duration = 0
            try:
                for step in range(max_num_timesteps):
                    self.duration += 1
                    action_index = self.ft_choose_action(self.states, self.epsilon)
                    len_before = len(self.game.snake)
                    code, is_over = self.game.ft_move_snake(self.ACTIONS[action_index], "off")
                    next_state = self.ft_encode_vision(self.game.ft_get_vision_snake())
                    len_after = len(self.game.snake)
                    event_code = code
                    if not is_over:
                        if len_after > len_before:
                            event_code = self.GREEN_APPLE
                            green_hits_win += (len_after - len_before)
                        elif len_after < len_before:
                            event_code = self.RED_APPLE
                            red_hits_win += (len_before - len_after)
                    reward = self.ft_calculate_reward(event_code, is_over, next_state)
                    
                    # Récompenses bonus pour la croissance
                    if len_after > max_length_global:
                        reward += 20.0  # Bonus pour nouveau record
                        max_length_global = len_after
                    if len_after > len_before:
                        reward += 10.0  # Bonus pour croissance
                    elif len_after < len_before:
                        reward -= 15.0  # Pénalité pour réduction
                    
                    self.ft_store_experience(self.states, action_index, reward, next_state, is_over)
                    if self.ft_check_update_conditions(step):
                        self.loss = self.ft_agent_learn()
                    self.states = next_state.copy()
                    if len_after > max_length_global:
                        max_length_global = len_after
                        self.len_hist.append(max_length_global)
                    if is_over:
                        break
            except Exception:
                self.is_over = True
                sys.exit(1)

            if self.duration > max_duration_global:
                max_duration_global = self.duration

            # Décroissance de l'epsilon après chaque session
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            if session_id % 100 == 0:
                max_len = max(self.len_hist) if self.len_hist else 0
                print(f"[REWARDS/100] sess={session_id} max_length={max_len} epsilon={self.epsilon} green={green_hits_win} red={red_hits_win}")
                green_hits_win = 0
                red_hits_win = 0

            if session_id % 500 == 0:
                print(f"=== SESSION {session_id} END ===")
                self.q_network.save_weights(f'./models/{session_id}sess.weights.h5')

            self.game.ft_reset()

        return max_length_global, max_duration_global

    def ft_reset(self):
        self.__init__(self.game)

def main():
    size = 10
    game = Game(size, size)
    networks = DeepQNetwork(game)
    try:
        sessions = 400
        max_length, max_duration = networks.ft_interaction(sessions)
        print(f"Game over, max length: {max_length}, max_duration: {max_duration}")
        networks.q_network.save_weights('./standard.weights.h5')
    except KeyboardInterrupt:
        del game
        del networks
        sys.exit(1)

if __name__ == "__main__":
    main()




# (optionnel) charger un checkpoint si tu veux reprendre :
        # try:
        #     self.q_network.load_weights('./models/600sess.weights.h5')
        #     print("[info] Loaded weights from ./models/600sess.weights.h5")
        #     self.epsilon = 0.2   # si reprise d'entraînement (sinon laisse 1.0)
        # except Exception:
        #     print("[info] No existing weights found, starting fresh.")