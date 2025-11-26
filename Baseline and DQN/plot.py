import numpy as np
import matplotlib.pyplot as plt
from typing import List

from DQN import ExperimentResult

def plot_avg_return(results: List[ExperimentResult]):
    """Plot moving-average return curves for one or more experiments.

    results: a list of ExperimentResult objects, e.g.
        [result_dqn, result_double, result_dueling]
    """
    WINDOW = results[0].config.WINDOW
    plt.figure(figsize=(8, 5))

    for res in results:
        # 1. Calculate the rolling average
        # Use np.convolve to calculate the rolling mean.
        # 'valid' mode means only points where the window fully overlaps are calculated,
        # so the resulting array is shorter by (WINDOW - 1).
        weights = np.ones(WINDOW) / WINDOW
        moving_avg_returns = np.convolve(res.returns, weights, mode='valid')
        
        steps = range(len(moving_avg_returns))
        label = f"{res.variant}"
        plt.plot(steps, moving_avg_returns, label=label)

    plt.xlabel("Episode")
    plt.ylabel(f"Moving average return (window={WINDOW})")
    plt.title("Average return curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

