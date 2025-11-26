import random
import numpy as np
import torch

from typing import List, Tuple

from tqdm import tqdm

import gymnasium as gym

from config import DQNConfig, ExperimentResult
from DQN import BaseDQNAgent, DQNAgent, DoubleDQNAgent, DuelingDQNAgent

# ----------------------
# Training loop (no main, to be called from notebook)
# ----------------------


def train_loop(
    config: DQNConfig,
    *,
    exploration: str = "epsilon",  # "epsilon" | "boltzmann" | "none"
    variant: str = "dqn",          # just a label for logging (e.g. "dqn", "double", "dueling")
    output_mode: str = "silent",   # "silent" | "format_print" | “progress_bar"
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

    env = gym.make("LunarLander-v3")

    # Initialize environment
    s, _ = env.reset(seed=config.SEED)
    episode_return = 0.0
    episode_len = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Choose agent class based on variant label. This makes it easy to
    # swap in DoubleDQNAgent or DuelingDQNAgent later.
    if variant.lower().startswith("dqn"):
        agent_cls = DQNAgent
    elif variant.lower().startswith("double"):
        agent_cls = DoubleDQNAgent
    elif variant.lower().startswith("dueling"):
        agent_cls = DuelingDQNAgent
    else:
        # Fallback to standard DQN if an unknown variant is passed.
        agent_cls = DQNAgent
    
    agent = agent_cls(obs_dim, n_actions, config, device=device)

    episode_lengths: List[int] = []
    returns: List[float] = []
    losses: List[float] = []
    grad_norms: List[float] = []
    q_means: List[float] = []
    q_stds: List[float] = []

    step_range = range(1, config.MAX_ENV_STEPS + 1)
    if output_mode == "progress_bar":
        step_range = tqdm(step_range)

    for global_step in step_range:
        # 1) Select action with exploration
        a = agent.select_action(s, global_step, exploration=exploration)

        # 2) Step environment
        s_next, r, terminated, truncated, info = env.step(a)
        # if truncated:
        #     r += -100
        # prev_y, current_y = s[1], s_next[1]
        # r += 0.03 * (prev_y - current_y)
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

            ep = len(returns)
            if output_mode == "format_print" and (ep % config.PRINT_EVERY == 0 or ep == 1):
                latest_avg_return = np.mean(returns[-config.WINDOW:])
                latest_avg_length = np.mean(episode_lengths[-config.WINDOW:])
                print(f"Episode: {ep:4d} | Global Step: {global_step:8,d} | The latest {config.WINDOW} avg return: {latest_avg_return:>8.1f} | avg length: {latest_avg_length:>6.1f}")
            # If not, reset episode
            s, _ = env.reset()
            episode_return = 0.0
            episode_len = 0
        else:
            s = s_next

    env.close()

    result = ExperimentResult(
        config=config,
        variant=variant,
        returns=returns,
        episode_lengths=episode_lengths,
        losses=losses,
        grad_norms=grad_norms,
        q_means=q_means,
        q_stds=q_stds,
    )

    return result, agent
