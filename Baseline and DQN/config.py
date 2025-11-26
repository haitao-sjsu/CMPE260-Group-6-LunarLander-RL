from dataclasses import dataclass
from typing import List, Tuple

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
    LR: float = 4e-4
    GRAD_CLIP_NORM: float = 5.0

    # Replay Buffer
    CAPACITY: int = 100_000
    TRAIN_START_SIZE: int = 5_000

    # Training Loop
    BATCH_SIZE: int = 128
    TARGET_UPDATE_FREQ: int = 1_000  # in environment steps
    MAX_ENV_STEPS: int = 500_000
    TAU = 0.01

    # Exploration: epsilon-greedy
    EPS_START: float = 1.0
    EPS_END: float = 0.01
    EPS_DECAY_STEPS: int = 100_000

    # Exploration: Boltzmann (optional)
    T_START: float = 1.0
    T_END: float = 0.01
    T_DECAY_STEPS: int = 100_000

    # Network Architecture
    HIDDEN_SIZES: Tuple[int, int] = (64, 64)

    #
    WINDOW: int = 100
    SOLVE_AT: int = 250
    PRINT_EVERY: int = 200

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
    losses: List[float]
    grad_norms: List[float]
    q_means: List[float]
    q_stds: List[float]


