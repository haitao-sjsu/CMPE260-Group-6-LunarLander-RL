from dataclasses import dataclass
from typing import List, Tuple, Optional

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym


# ----------------------
# Config
# ----------------------


@dataclass
class DQNConfig:
    """Configuration for DQN and its variants.

    This is shared across different experiments so we can easily tweak
    hyperparameters like LR, HIDDEN_SIZES, etc.
    """

    # Reproducibility
    SEED: int = 42

    # Discounting
    GAMMA: float = 0.99

    # Optimization
    LR: float = 1e-4
    GRAD_CLIP_NORM: float = 10.0

    # Replay Buffer
    CAPACITY: int = 100_000
    TRAIN_START_SIZE: int = 1_000

    # Training Loop
    BATCH_SIZE: int = 64
    TARGET_UPDATE_FREQ: int = 1_000  # in environment steps
    MAX_ENV_STEPS: int = 100_000

    # Exploration: epsilon-greedy
    EPS_START: float = 1.0
    EPS_END: float = 0.05
    EPS_DECAY_STEPS: int = 50_000

    # Exploration: Boltzmann (optional)
    T_START: float = 1.0
    T_END: float = 0.05
    T_DECAY_STEPS: int = 50_000

    # Network Architecture
    HIDDEN_SIZES: Tuple[int, int] = (32, 32)


# ----------------------
# Replay Buffer
# ----------------------


class ReplayBuffer:
    """Simple replay buffer for off-policy DQN.

    Stores transitions and supports uniform random sampling.
    """

    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.size = 0
        self.ptr = 0

        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def add(self, s, a, r, s_next, done):
        idx = self.ptr
        self.states[idx] = s
        self.actions[idx] = a
        self.rewards[idx] = r
        self.next_states[idx] = s_next
        self.dones[idx] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def __len__(self) -> int:
        return self.size

    def sample(self, batch_size: int, device: torch.device):
        idxs = np.random.randint(0, self.size, size=batch_size)

        S = torch.as_tensor(self.states[idxs], dtype=torch.float32, device=device)
        A = torch.as_tensor(self.actions[idxs], dtype=torch.int64, device=device)
        R = torch.as_tensor(self.rewards[idxs], dtype=torch.float32, device=device)
        S_next = torch.as_tensor(self.next_states[idxs], dtype=torch.float32, device=device)
        D = torch.as_tensor(self.dones[idxs], dtype=torch.float32, device=device)
        return S, A, R, S_next, D


# ----------------------
# Q Network
# ----------------------


class QNetwork(nn.Module):
    """Simple MLP Q-network for discrete action spaces."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes: Tuple[int, int]):
        super().__init__()
        h1, h2 = hidden_sizes
        self.net = nn.Sequential(
            nn.Linear(obs_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ----------------------
# Experiment result structure
# ----------------------


@dataclass
class ExperimentResult:
    """Container for logging training statistics.

    This makes it easy to compare different algorithms / hyperparameters
    and to plot learning curves later.
    """

    config: DQNConfig
    variant: str
    episode_lengths: List[int]
    returns: List[float]
    moving_avg_returns: List[float]
    losses: List[float]
    grad_norms: List[float]
    q_means: List[float]
    q_stds: List[float]
    steps_to_200: Optional[int]


# ----------------------
# DQN Agent base class and variants (online + target network, replay, exploration)
# ----------------------


class BaseDQNAgent:
    """Base class for DQN-style agents.

    This class implements standard DQN with:
    - online and target Q-networks
    - experience replay
    - epsilon-greedy or Boltzmann exploration

    Variants such as Double DQN and Dueling DQN can be implemented by
    subclassing this base class and overriding specific methods such as
    `compute_targets` or the network architecture.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        config: DQNConfig,
        device: Optional[torch.device] = None,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Q networks (subclasses can override these attributes if needed)
        self.q_online = QNetwork(obs_dim, n_actions, config.HIDDEN_SIZES).to(self.device)
        self.q_target = QNetwork(obs_dim, n_actions, config.HIDDEN_SIZES).to(self.device)
        self.q_target.load_state_dict(self.q_online.state_dict())
        self.q_target.eval()

        self.optimizer = torch.optim.Adam(self.q_online.parameters(), lr=config.LR)

        # replay buffer
        self.replay = ReplayBuffer(config.CAPACITY, obs_dim)

    # ------------- Exploration schedules -------------

    def epsilon_at_step(self, step: int) -> float:
        """Linear epsilon decay from EPS_START to EPS_END."""
        cfg = self.config
        if step >= cfg.EPS_DECAY_STEPS:
            return cfg.EPS_END
        frac = step / float(cfg.EPS_DECAY_STEPS)
        return cfg.EPS_START + frac * (cfg.EPS_END - cfg.EPS_START)

    def temperature_at_step(self, step: int) -> float:
        """Linear temperature decay from T_START to T_END."""
        cfg = self.config
        if step >= cfg.T_DECAY_STEPS:
            return cfg.T_END
        frac = step / float(cfg.T_DECAY_STEPS)
        return cfg.T_START + frac * (cfg.T_END - cfg.T_START)

    # ------------- Action selection -------------

    def select_action(self, s_np: np.ndarray, step: int, exploration: str = "epsilon") -> int:
        """Select an action given a state using the chosen exploration strategy."""
        s_t = torch.as_tensor(s_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_online(s_t)  # shape (1, n_actions)

        if exploration == "none":
            # Greedy action
            return int(torch.argmax(q_values, dim=-1).item())
        elif exploration == "epsilon":
            eps = self.epsilon_at_step(step)
            if random.random() < eps:
                return random.randrange(self.n_actions)
            return int(torch.argmax(q_values, dim=-1).item())
        elif exploration == "boltzmann":
            T = self.temperature_at_step(step)
            # Softmax over Q/T
            logits = q_values[0] / max(T, 1e-6)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            return int(np.random.choice(self.n_actions, p=probs))
        else:
            raise ValueError(f"Unknown exploration strategy: {exploration}")

    def select_deterministic_action(self, s_np: np.ndarray) -> int:
        """Select a greedy action without exploration.

        This is intended for evaluation, where we typically want the
        deterministic policy corresponding to the current Q-network.
        """
        s_t = torch.as_tensor(s_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_online(s_t)
            action = int(torch.argmax(q_values, dim=-1).item())
        return action

  # ------------- Target computation & update -------------

    def compute_targets(self, S_next: torch.Tensor, R: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
        """Compute bootstrapped targets using the target network.

        Standard DQN target:
        y = r + gamma * (1 - done) * max_a' Q_target(s', a')

        Subclasses (e.g. Double DQN) can override this method.
        """
        with torch.no_grad():
            Qn = self.q_target(S_next)  # (batch, n_actions)
            max_next = Qn.max(dim=1).values  # (batch,)
            y = R + self.config.GAMMA * (1.0 - D) * max_next
        return y

    def update(self, global_step: int) -> Optional[Tuple[float, float, float, float]]:
        """Sample a batch from replay and perform one gradient step.

        Returns a tuple of (loss, grad_norm, q_mean, q_std) for logging.
        If replay is not large enough yet, returns None.
        """
        cfg = self.config
        if len(self.replay) < cfg.TRAIN_START_SIZE:
            return None

        S, A, R, S_next, D = self.replay.sample(cfg.BATCH_SIZE, self.device)

        # Current Q(s, a)
        Q_all = self.q_online(S)  # (batch, n_actions)
        Q_sa = Q_all.gather(1, A.view(-1, 1)).squeeze(1)  # (batch,)

        # Targets (can be overridden in subclasses)
        y = self.compute_targets(S_next, R, D)

        # Loss
        loss = F.smooth_l1_loss(Q_sa, y)

        # Backprop
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.q_online.parameters(), cfg.GRAD_CLIP_NORM)
        self.optimizer.step()

        # Hard update target network
        if global_step % cfg.TARGET_UPDATE_FREQ == 0:
            self.q_target.load_state_dict(self.q_online.state_dict())

        # Some statistics for logging
        with torch.no_grad():
            q_mean = Q_all.mean().item()
            q_std = Q_all.std().item()

        return float(loss.item()), float(grad_norm), q_mean, q_std


class DQNAgent(BaseDQNAgent):
    """Standard DQN agent.

    Inherits all behavior from BaseDQNAgent without modification.
    """

    pass


class DoubleDQNAgent(BaseDQNAgent):
    """Skeleton for Double DQN agent.

    For now, this behaves the same as standard DQN. To implement true
    Double DQN, override `compute_targets` to use the online network
    for action selection and the target network for evaluation.
    """

    # TODO: override compute_targets for Double DQN behavior.
    pass


class DuelingDQNAgent(BaseDQNAgent):
    """Skeleton for Dueling DQN agent.

    For now, this uses the same architecture as the base class. To
    implement true Dueling DQN, replace `self.q_online` and
    `self.q_target` with dueling-style networks in `__init__`.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        config: DQNConfig,
        device: Optional[torch.device] = None,
    ):
        super().__init__(obs_dim, n_actions, config, device=device)


# ----------------------
# Training loop (no main, to be called from notebook)
# ----------------------


def train_loop(
    env: gym.Env,
    config: DQNConfig,
    *,
    exploration: str = "epsilon",  # "epsilon" | "boltzmann" | "none"
    variant: str = "dqn",          # just a label for logging (e.g. "dqn", "double", "dueling")
) -> Tuple[ExperimentResult, BaseDQNAgent]:
    """Main training loop for DQN on a Gymnasium environment.

    This version always uses experience replay and a target network.
    It logs several statistics that will be useful for later analysis
    and visualization in your project.
    """

    # Set seeds for reproducibility
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if hasattr(env, "reset"):
        env.reset(seed=config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Choose agent class based on variant label. This makes it easy to
    # swap in DoubleDQNAgent or DuelingDQNAgent later.
    if variant.lower() == "dqn":
        agent_cls: Type[BaseDQNAgent] = DQNAgent
    elif variant.lower() == "double":
        agent_cls = DoubleDQNAgent
    elif variant.lower() == "dueling":
        agent_cls = DuelingDQNAgent
    else:
        # Fallback to standard DQN if an unknown variant is passed.
        agent_cls = DQNAgent
    
    agent = agent_cls(obs_dim, n_actions, config, device=device)

    episode_lengths: List[int] = []
    returns: List[float] = []
    moving_avg_returns: List[float] = []
    losses: List[float] = []
    grad_norms: List[float] = []
    q_means: List[float] = []
    q_stds: List[float] = []

    # For moving average and steps-to-200
    from collections import deque

    window = deque(maxlen=100)
    steps_to_200: Optional[int] = None

    # Initialize environment
    s, _ = env.reset(seed=config.SEED)
    episode_return = 0.0
    episode_len = 0

    for global_step in range(1, config.MAX_ENV_STEPS + 1):
        # 1) Select action with exploration
        a = agent.select_action(s, global_step, exploration=exploration)

        # 2) Step environment
        s_next, r, terminated, truncated, info = env.step(a)
        done = bool(terminated or truncated)

        # 3) Store transition in replay buffer
        agent.replay.add(s.astype(np.float32), a, float(r), s_next.astype(np.float32), done)

        # 4) Accumulate return and length
        episode_return += r
        episode_len += 1

        # 5) DQN update (if enough data collected)
        update_info = agent.update(global_step)
        if update_info is not None:
            loss, grad_norm, q_mean, q_std = update_info
            losses.append(loss)
            grad_norms.append(grad_norm)
            q_means.append(q_mean)
            q_stds.append(q_std)

        # 6) Episode end handling
        if done:
            episode_lengths.append(episode_len)
            returns.append(episode_return)
            window.append(episode_return)
            moving_avg_returns.append(float(np.mean(window)))

            # Check if we have solved the task (avg return >= 200 over last 100 eps)
            if len(window) == 100 and steps_to_200 is None:
                if np.mean(window) >= 200.0:
                    steps_to_200 = global_step

            # Reset episode
            s, _ = env.reset()
            episode_return = 0.0
            episode_len = 0
        else:
            s = s_next

    result = ExperimentResult(
        config=config,
        variant=variant,
        returns=returns,
        episode_lengths=episode_lengths,
        moving_avg_returns=moving_avg_returns,
        losses=losses,
        grad_norms=grad_norms,
        q_means=q_means,
        q_stds=q_stds,
        steps_to_200=steps_to_200,
    )

    return result, agent
