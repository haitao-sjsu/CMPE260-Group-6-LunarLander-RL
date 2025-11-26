from typing import Tuple, Optional

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DQNConfig

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


class DuelingQNetwork(nn.Module):
    """Dueling architecture for discrete actions: shared torso + (Value, Advantage) heads."""
    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes: Tuple[int, int]):
        super().__init__()
        h1, h2 = hidden_sizes

        # Shared feature extractor
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
        )

        # Value and Advantage heads
        self.value = nn.Linear(h2, 1)
        self.advantage = nn.Linear(h2, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.feature(x)                      # (batch, h2)
        v = self.value(z)                        # (batch, 1)
        a = self.advantage(z)                    # (batch, n_actions)

        # Subtract mean advantage for identifiability/stability
        a_mean = a.mean(dim=1, keepdim=True)     # (batch, 1)
        q = v + (a - a_mean)                     # (batch, n_actions)
        return q


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

    def select_deterministic_action(self, env, s_np: np.ndarray) -> int:
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

    def target_network_update_(self, global_step):
        if global_step % self.config.TARGET_UPDATE_FREQ == 0:
            self.q_target.load_state_dict(self.q_online.state_dict())

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
        self.target_network_update_(global_step)

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
    """Double DQN agent.

    Key idea:
      - Use the *online* network to select the next action (argmax).
      - Use the *target* network to evaluate that chosen action.
    This decouples action selection from action evaluation and reduces
    the overestimation bias present in standard DQN.

    Target:
      y = r + gamma * (1 - done) * Q_target(s', argmax_a Q_online(s', a))
    """

    def compute_targets(
        self,
        S_next: torch.Tensor,
        R: torch.Tensor,
        D: torch.Tensor
    ) -> torch.Tensor:
        """Compute Double DQN bootstrap targets.

        Args:
            S_next: Next states, shape (batch, obs_dim).
            R: Rewards, shape (batch,).
            D: Done flags in {0,1}, shape (batch,).

        Returns:
            y: Target values for Q(s,a), shape (batch,).
        """
        with torch.no_grad():
            # 1) Action selection with the *online* network: a* = argmax_a Q_online(s', a)
            q_next_online = self.q_online(S_next)                      # (batch, n_actions)
            next_actions = torch.argmax(q_next_online, dim=1, keepdim=True)  # (batch, 1)

            # 2) Action evaluation with the *target* network: Q_target(s', a*)
            q_next_target = self.q_target(S_next)                      # (batch, n_actions)
            q_selected = q_next_target.gather(1, next_actions).squeeze(1)     # (batch,)

            # 3) Double DQN target
            y = R + self.config.GAMMA * (1.0 - D) * q_selected

        return y

            
class DuelingDQNAgent(BaseDQNAgent):
    """Dueling DQN: replace online/target networks with dueling heads.
    Training loop, replay, target updates, and targets stay the same.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        config: DQNConfig,
        device: Optional[torch.device] = None,
    ):
        # Initialize everything as usual, then replace the networks and optimizer.
        super().__init__(obs_dim, n_actions, config, device=device)

        # Swap in dueling networks
        self.q_online = DuelingQNetwork(obs_dim, n_actions, config.HIDDEN_SIZES).to(self.device)
        self.q_target = DuelingQNetwork(obs_dim, n_actions, config.HIDDEN_SIZES).to(self.device)

        # Hard sync target and set optimizer on the new parameters
        self.q_target.load_state_dict(self.q_online.state_dict())
        self.q_target.eval()
        self.optimizer = torch.optim.Adam(self.q_online.parameters(), lr=config.LR)