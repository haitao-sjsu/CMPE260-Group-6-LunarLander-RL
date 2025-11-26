from dataclasses import dataclass
from typing import List, Tuple, Optional, Type
from collections import deque

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym


# ----------------------
# Config
# ----------------------


@dataclass
class PPOConfig:
    """Configuration for PPO and its variants."""

    # Reproducibility
    SEED: int = 42

    # Discounting
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95

    # PPO Specific
    CLIP_EPSILON: float = 0.2
    PPO_EPOCHS: int = 10
    PPO_BATCH_SIZE: int = 64
    ENTROPY_COEFF: float = 0.01

    # Network Architecture
    HIDDEN_SIZES: Tuple[int, int] = (32, 32)

    # Optimization
    LR_ACTOR: float = 3e-4
    LR_CRITIC: float = 1e-3
    MAX_GRAD_NORM: float = 0.5

    # Training Loop
    STEPS_PER_EPOCH: int = 2048
    MAX_ENV_STEPS: int = 200_000


# ----------------------
# Actor and Critic Networks
# ----------------------


class Actor(nn.Module):
    """MLP Actor network for discrete action spaces (outputs logits)."""
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
        """Forward pass to get action logits."""
        return self.net(x)


class Critic(nn.Module):
    """MLP Critic network (outputs state value)."""
    def __init__(self, obs_dim: int, hidden_sizes: Tuple[int, int]):
        super().__init__()
        h1, h2 = hidden_sizes
        self.net = nn.Sequential(
            nn.Linear(obs_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1), # Output a single scalar value for the state value
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to get state value."""
        return self.net(x)


# ----------------------
# PPO Agent
# ----------------------


class PPOAgent:
    """PPO Agent class."""

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        config: PPOConfig,
        device: torch.device,
    ):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.config = config
        self.device = device

        # Actor and Critic networks
        self.actor = Actor(obs_dim, n_actions, config.HIDDEN_SIZES).to(device)
        self.critic = Critic(obs_dim, config.HIDDEN_SIZES).to(device)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.LR_ACTOR)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.LR_CRITIC)

    def select_action(self, s_np: np.ndarray) -> Tuple[int, float, float]:
        """Selects an action, its log_prob, and the entropy of the action distribution."""
        s_t = torch.as_tensor(s_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.actor(s_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
        return action.item(), log_prob.item(), entropy.item()

    def select_deterministic_action(self, s_np: np.ndarray) -> int:
        """Selects a greedy action without exploration."""
        s_t = torch.as_tensor(s_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.actor(s_t)
            action = torch.argmax(logits, dim=-1)
        return action.item()

    def compute_advantages_and_returns(self,
                                       rewards: List[float],
                                       values: List[float],
                                       dones: List[bool],
                                       last_value: float
                                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes Generalized Advantage Estimation (GAE) and discounted returns."""
        cfg = self.config
        advantages = []
        returns = []

        # Convert to tensors
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        values_t = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        # Add the last value for GAE calculation
        all_values = torch.cat([values_t, torch.as_tensor([last_value], device=self.device)])

        # GAE calculation (from back to front)
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards_t[i] + cfg.GAMMA * all_values[i + 1] * (1 - dones_t[i]) - all_values[i]
            gae = delta + cfg.GAMMA * cfg.GAE_LAMBDA * (1 - dones_t[i]) * gae
            advantages.insert(0, gae)
        advantages_t = torch.stack(advantages)

        # Discounted returns (targets for the critic)
        returns_t = advantages_t + values_t

        return advantages_t, returns_t


    def update(self,
               states: torch.Tensor,
               actions: torch.Tensor,
               log_probs_old: torch.Tensor,
               advantages: torch.Tensor,
               returns: torch.Tensor):
        """Performs PPO network updates."""
        cfg = self.config

        # Normalize advantages for more stable training
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        data_size = states.size(0)
        indices = np.arange(data_size)

        for _ in range(cfg.PPO_EPOCHS):
            np.random.shuffle(indices)
            for start_idx in range(0, data_size, cfg.PPO_BATCH_SIZE):
                end_idx = min(start_idx + cfg.PPO_BATCH_SIZE, data_size)
                batch_indices = indices[start_idx:end_idx]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_log_probs_old = log_probs_old[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Critic update
                self.critic_optimizer.zero_grad()
                values_pred = self.critic(batch_states).squeeze(-1)
                critic_loss = F.mse_loss(values_pred, batch_returns)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.MAX_GRAD_NORM)
                self.critic_optimizer.step()

                # Actor update
                self.actor_optimizer.zero_grad()
                logits_new = self.actor(batch_states)
                dist_new = Categorical(logits=logits_new)
                log_probs_new = dist_new.log_prob(batch_actions)

                # PPO ratio
                ratio = torch.exp(log_probs_new - batch_log_probs_old)

                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - cfg.CLIP_EPSILON, 1.0 + cfg.CLIP_EPSILON) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Entropy bonus
                entropy_loss = -cfg.ENTROPY_COEFF * dist_new.entropy().mean()

                # Total actor loss
                total_actor_loss = actor_loss + entropy_loss
                total_actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.MAX_GRAD_NORM)
                self.actor_optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": dist_new.entropy().mean().item(),
            "clip_fraction": (ratio.abs() > (1 + cfg.CLIP_EPSILON)).float().mean().item() # Approx clip frac
        }


# ----------------------
# Experiment result structure for PPO
# ----------------------


@dataclass
class PPOExperimentResult:
    """Container for logging PPO training statistics."""

    config: PPOConfig
    variant: str
    episode_lengths: List[int]
    returns: List[float]
    moving_avg_returns: List[float]
    actor_losses: List[float]
    critic_losses: List[float]
    entropies: List[float]
    clip_fractions: List[float]
    steps_to_200: Optional[int]


def train_loop_ppo(
        env: gym.Env,
        config: PPOConfig,
        *,
        variant: str = "ppo",  # just a label for logging
) -> Tuple[PPOExperimentResult, PPOAgent]:
    """Main training loop for PPO on a Gymnasium environment."""

    # Set seeds for reproducibility
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if hasattr(env, "reset"):
        env.reset(seed=config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = PPOAgent(obs_dim, n_actions, config, device=device)

    episode_lengths: List[int] = []
    returns: List[float] = []
    moving_avg_returns: List[float] = []
    actor_losses: List[float] = []
    critic_losses: List[float] = []
    entropies: List[float] = []
    clip_fractions: List[float] = []

    # For moving average and steps-to-200
    window = deque(maxlen=100)
    steps_to_200: Optional[int] = None

    # Initialize environment
    s, _ = env.reset(seed=config.SEED)
    episode_return = 0.0
    episode_len = 0

    global_step = 0

    while global_step < config.MAX_ENV_STEPS:
        # Lists to collect data for the current rollout
        rollout_states: List[np.ndarray] = []
        rollout_actions: List[int] = []
        rollout_log_probs: List[float] = []
        rollout_rewards: List[float] = []
        rollout_values: List[float] = []
        rollout_dones: List[bool] = []

        for _ in range(config.STEPS_PER_EPOCH):
            global_step += 1

            # Get current state's value and select action
            s_tensor = torch.as_tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                value = agent.critic(s_tensor).item()
            a, log_prob, entropy = agent.select_action(s)

            # Step environment
            s_next, r, terminated, truncated, info = env.step(a)
            done = bool(terminated or truncated)

            # Store transition data
            rollout_states.append(s.astype(np.float32))
            rollout_actions.append(a)
            rollout_log_probs.append(log_prob)
            rollout_rewards.append(r)
            rollout_values.append(value)
            rollout_dones.append(done)

            # Accumulate return and length for logging
            episode_return += r
            episode_len += 1

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

            if global_step >= config.MAX_ENV_STEPS:  # Stop if max steps reached during rollout collection
                break

        # After collecting STEPS_PER_EPOCH (or less if MAX_ENV_STEPS reached):
        if len(rollout_rewards) == 0:  # Handle case where no steps were collected in the last epoch due to early exit
            continue

        # Get last_value for GAE calculation
        if done:
            last_value = 0.0
        else:
            s_last_tensor = torch.as_tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                last_value = agent.critic(s_last_tensor).item()

        # Compute advantages and returns
        advantages_t, returns_t = agent.compute_advantages_and_returns(
            rollout_rewards, rollout_values, rollout_dones, last_value
        )

        # Convert collected lists to tensors for PPO update
        states_t = torch.as_tensor(np.array(rollout_states), dtype=torch.float32, device=device)
        actions_t = torch.as_tensor(np.array(rollout_actions), dtype=torch.int64, device=device)
        log_probs_t = torch.as_tensor(np.array(rollout_log_probs), dtype=torch.float32, device=device)

        # Perform PPO updates
        update_info = agent.update(
            states_t, actions_t, log_probs_t, advantages_t, returns_t
        )

        actor_losses.append(update_info["actor_loss"])
        critic_losses.append(update_info["critic_loss"])
        entropies.append(update_info["entropy"])
        clip_fractions.append(update_info["clip_fraction"])

    result = PPOExperimentResult(
        config=config,
        variant=variant,
        returns=returns,
        episode_lengths=episode_lengths,
        moving_avg_returns=moving_avg_returns,
        actor_losses=actor_losses,
        critic_losses=critic_losses,
        entropies=entropies,
        clip_fractions=clip_fractions,
        steps_to_200=steps_to_200,
    )

    return result, agent