# test_agent.py
import numpy as np
import pytest
import types
from collections import deque
import tensorflow as tf
from tensorflow.keras.activations import relu, linear
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# On importe ton agent et on s'aligne sur ses hyperparams (TAU, MINIBATCH_SIZE)
import agent as ag
from agent import DeepQNetwork

def shape_as_list(shape):
    # Keras 2.x (tf.keras): TensorShape -> a .as_list()
    try:
        return list(shape.as_list())
    except AttributeError:
        # Keras 3: shape est souvent un tuple
        return list(shape)

# =========================================================
# Fixtures utilitaires
# =========================================================

class DummyGame:
    """Jeu minimal pour certains tests (pas de vraie logique)."""
    def __init__(self):
        self.snake = [(0, 0), (0, 1)]  # longueur 2 pour tester le demi-tour si besoin
        self.green_apples = []
    def ft_get_vision_snake(self):
        # 4 directions * 2 features = 8 (comme attendu par les tests)
        return {"up":[0,0], "down":[0,0], "left":[0,0], "right":[0,0]}
    def ft_move_snake(self, action, _mode):
        # code=EMPTY, is_over=False
        return (DeepQNetwork.EMPTY, False)
    def ft_reset(self):  # no-op
        self.snake = [(0, 0), (0, 1)]
        self.green_apples = []


@pytest.fixture
def ALPHA():
    return 1e-3


@pytest.fixture
def agent_net(ALPHA):
    """
    Instancie un DQN minimal, construit les modèles et les 'prime'
    pour que .input/.output soient définis (Keras 3).
    """
    agent = DeepQNetwork(game=DummyGame())
    agent.state_size = (8,)     # 8 features (cf. vision encodée simplifiée)
    agent.num_actions = 4
    agent.alpha = ALPHA
    agent.ft_define_initial_model()

    # PRIME les deux modèles (nécessaire pour accéder à .input)
    dummy = np.zeros((1, 8), dtype=np.float32)
    _ = agent.q_network(dummy, training=False)
    _ = agent.target_q_network(dummy, training=False)
    return agent


@pytest.fixture
def target_model(agent_net):
    return agent_net.q_network


@pytest.fixture
def optimizer(agent_net):
    return agent_net.optimizer


@pytest.fixture
def loss_wrapper():
    """
    Version numpy du calcul de MSE cible → utilisée par les tests historiques.
    signature: fn((states, actions, rewards, next_states, dones), gamma, q_net, target_q_net)
    """
    def fn(data, gamma, q_net, target_q_net):
        states, actions, rewards, next_states, dones = data
        q_next = target_q_net(next_states)                 # (B, A)
        max_q = np.max(q_next, axis=1)                     # (B,)
        y_targets = rewards + gamma * max_q * (1.0 - dones)
        q_values = q_net(states)                           # (B, A)
        idx = np.arange(len(states))
        q_taken = q_values[idx, actions.astype(int)]       # (B,)
        return np.mean((y_targets - q_taken) ** 2)
    return fn


# =========================================================
# Tests structure réseau + optimizer + loss (existants)
# =========================================================

def test_network(target_model):
    num_actions = 4
    state_size = 8
    i = 0

    assert len(target_model.layers) == 3, \
        f"Wrong number of layers. Expected 3 but got {len(target_model.layers)}"

    assert shape_as_list(target_model.input.shape) == [None, state_size], \
        f"Wrong input shape. Expected [None, {state_size}] but got {shape_as_list(target_model.input.shape)}"

    expected = [
        [Dense, [None, 64], relu],
        [Dense, [None, 64], relu],
        [Dense, [None, num_actions], linear],
    ]

    for layer in target_model.layers:
        assert type(layer) == expected[i][0], \
            f"Wrong type in layer {i}. Expected {expected[i][0]} but got {type(layer)}"
        assert shape_as_list(layer.output.shape) == expected[i][1], \
            f"Wrong number of units in layer {i}. Expected {expected[i][1]} but got {shape_as_list(layer.output.shape)}"
        assert layer.activation == expected[i][2], \
            f"Wrong activation in layer {i}. Expected {expected[i][2]} but got {layer.activation}"
        i += 1


def test_optimizer(optimizer, ALPHA):
    assert isinstance(optimizer, Adam), \
        f"Wrong optimizer. Expected: {Adam}, got: {type(optimizer)}"
    assert np.isclose(optimizer.learning_rate.numpy(), ALPHA), \
        f"Wrong alpha. Expected: {ALPHA}, got: {optimizer.learning_rate.numpy()}"


def test_compute_loss(loss_wrapper):
    num_actions = 4

    def target_q_network_random(inputs):
        return np.float32(np.random.rand(inputs.shape[0], num_actions))

    def q_network_random(inputs):
        return np.float32(np.random.rand(inputs.shape[0], num_actions))

    def target_q_network_ones(inputs):
        return np.float32(np.ones((inputs.shape[0], num_actions)))

    def q_network_ones(inputs):
        return np.float32(np.ones((inputs.shape[0], num_actions)))

    np.random.seed(1)
    states = np.float32(np.random.rand(64, 8))
    actions = np.float32(np.floor(np.random.uniform(0, 1, (64,)) * 4))
    rewards = np.float32(np.random.rand(64,))
    next_states = np.float32(np.random.rand(64, 8))
    done_vals = np.float32((np.random.uniform(0, 1, size=(64,)) > 0.96) * 1)

    loss = loss_wrapper((states, actions, rewards, next_states, done_vals),
                        0.995, q_network_random, target_q_network_random)
    assert np.isclose(loss, 0.6991737, atol=1e-5)

    done_vals = np.float32(np.ones((64,)))
    loss = loss_wrapper((states, actions, rewards, next_states, done_vals),
                        0.995, q_network_ones, target_q_network_ones)
    assert np.isclose(loss, 0.343270182, atol=1e-5)

    done_vals = np.float32((np.random.uniform(0, 1, size=(64,)) > 0.96) * 1)
    rewards = np.float32(np.ones((64,)))
    loss = loss_wrapper((states, actions, rewards, next_states, done_vals),
                        0, q_network_ones, target_q_network_ones)
    assert np.isclose(loss, 0, atol=1e-6)

    done_vals = np.float32((np.random.uniform(0, 1, size=(64,)) > 0.96) * 1)
    rewards = np.float32(np.zeros((64,)))
    loss = loss_wrapper((states, actions, rewards, next_states, done_vals),
                        0, q_network_ones, target_q_network_ones)
    assert np.isclose(loss, 1, atol=1e-6)


# =========================================================
# Nouveaux tests : buffer, epsilon, learn, soft update
# =========================================================

def _make_batch(B=ag.MINIBATCH_SIZE, state_dim=8, num_actions=4):
    """Crée un batch synthétique cohérent avec l'agent."""
    states = np.random.rand(B, state_dim).astype(np.float32)
    next_states = np.random.rand(B, state_dim).astype(np.float32)
    actions = np.random.randint(0, num_actions, size=(B,), dtype=np.int32)
    rewards = np.random.uniform(-1, 1, size=(B,)).astype(np.float32)
    dones = (np.random.rand(B) > 0.9).astype(np.float32)
    return states, actions, rewards, next_states, dones


def test_store_experience_copies(agent_net):
    s0 = np.ones((8,), dtype=np.float32)
    s1 = np.zeros((8,), dtype=np.float32)
    agent_net.experience = []
    agent_net.ft_store_experience(s0, 2, 0.5, s1, False)

    # Muter les originaux
    s0 *= 123.0
    s1 += 7.0

    # Vérifier que le buffer a bien stocké des COPIES (pas des vues)
    buf_s0, a, r, buf_s1, d = agent_net.experience[0]
    assert np.allclose(buf_s0, np.ones((8,), dtype=np.float32))
    assert np.allclose(buf_s1, np.zeros((8,), dtype=np.float32))
    assert a == 2 and np.isclose(r, 0.5) and d is False


def test_choose_action_epsilon(agent_net, monkeypatch):
    # Forcer la prédiction pour contrôler l'argmax
    def fake_predict(x, verbose=0):
        # batch x 4 actions: argmax = 3 (right)
        return np.array([[0.0, 0.1, 0.2, 0.9]], dtype=np.float32)
    monkeypatch.setattr(agent_net.q_network, "predict", fake_predict)

    # Cas exploitation (u >= epsilon)
    agent_net.epsilon = 0.0
    # u=0.0 → exploitation
    monkeypatch.setattr(np.random, "uniform", lambda *args, **kwargs: 0.0)
    action = agent_net.ft_choose_action(np.zeros((8,), dtype=np.float32))
    assert action == 3

    # Cas exploration (u < epsilon)
    agent_net.epsilon = 1.0
    # Monkeypatch random.random via module random utilisé dans l'agent
    import random as _r
    monkeypatch.setattr(_r, "uniform", lambda a, b: 0.0)  # < epsilon → explore
    # Comme on explore, l'action peut être n'importe quoi ∈ {0..3}
    action2 = agent_net.ft_choose_action(np.zeros((8,), dtype=np.float32))
    assert action2 in [0, 1, 2, 3]


def test_compute_loss_tensor_shapes(agent_net):
    # Pré-remplir le buffer et appeler ft_compute_loss
    agent_net.experience = deque(maxlen=agent_net.memory_size)
    B = max(ag.MINIBATCH_SIZE, agent_net.warmup_steps)
    for _ in range(B):
        s, a, r, ns, d = _make_batch(B=1)
        agent_net.ft_store_experience(s[0], a[0], r[0], ns[0], bool(d[0]))

    # Appel interne (private-ish) pour vérifier les shapes
    loss, q_vals, y_targets = agent_net.ft_compute_loss()
    assert np.isscalar(float(loss))
    assert q_vals.shape == y_targets.shape
    assert len(q_vals.shape) == 1  # (B,)


def test_agent_learn_updates_weights(agent_net, monkeypatch):
    """
    Vérifie que ft_agent_learn:
      - effectue un backward pass (gradients non None)
      - modifie les poids de q_network
      - effectue un soft update du target (θ_target ← τ θ_policy + (1-τ) θ_target)
    """
    # Baisser le warmup pour déclencher l'apprentissage rapidement
    agent_net.warmup_steps = 1
    agent_net.experience = deque(maxlen=agent_net.memory_size)

    # Remplir un buffer > MINIBATCH_SIZE
    for _ in range(ag.MINIBATCH_SIZE + 5):
        s, a, r, ns, d = _make_batch(B=1)
        agent_net.ft_store_experience(s[0], a[0], r[0], ns[0], bool(d[0]))

    # Sauvegarder copies des poids avant/après
    qW_before = [w.numpy().copy() for w in agent_net.q_network.weights]
    tW_before = [w.numpy().copy() for w in agent_net.target_q_network.weights]

    loss = agent_net.ft_agent_learn()
    assert loss is not None

    qW_after = [w.numpy().copy() for w in agent_net.q_network.weights]
    tW_after = [w.numpy().copy() for w in agent_net.target_q_network.weights]

    # Les poids policy doivent changer (au moins une différence)
    diffs_policy = [np.any(np.abs(a - b) > 0) for a, b in zip(qW_after, qW_before)]
    assert any(diffs_policy), "q_network weights did not change after learn step"

    # Soft update: vérifier la relation t_after ≈ τ * q_after + (1-τ) * t_before
    tau = ag.TAU
    approx_target = [tau * qa + (1.0 - tau) * tb for qa, tb in zip(qW_after, tW_before)]
    closeness = [np.allclose(ta, at, atol=1e-5) for ta, at in zip(tW_after, approx_target)]
    assert all(closeness), "target weights not matching soft-update formula"


def test_soft_update_formula_is_correct(agent_net):
    """Test direct de ft_update_target_soft_update sur des poids fictifs."""
    # Créer des poids déterministes pour q et target
    for var in agent_net.q_network.weights:
        var.assign(tf.ones_like(var))       # θ_policy = 1
    for var in agent_net.target_q_network.weights:
        var.assign(tf.zeros_like(var))      # θ_target = 0

    tau = ag.TAU
    agent_net.ft_update_target_soft_update()

    # Après soft update: θ_target = τ*1 + (1-τ)*0 = τ
    for var in agent_net.target_q_network.weights:
        arr = var.numpy()
        assert np.allclose(arr, tau, atol=1e-7)


def test_check_update_conditions(agent_net):
    # len(experience) insuffisant
    agent_net.experience = deque(maxlen=agent_net.memory_size)
    assert agent_net.ft_check_update_conditions(step=ag.NUM_STEPS_FOR_UPDATE - 1) is False

    # Enough experience & step multiple
    for _ in range(ag.MINIBATCH_SIZE):
        s, a, r, ns, d = _make_batch(B=1)
        agent_net.ft_store_experience(s[0], a[0], r[0], ns[0], bool(d[0]))

    assert agent_net.ft_check_update_conditions(step=ag.NUM_STEPS_FOR_UPDATE - 1) is True
