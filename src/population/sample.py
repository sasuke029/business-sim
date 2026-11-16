"""Draw individual agents from a fitted joint distribution."""
import numpy as np
import pandas as pd

from .schema import categories


def sample_agents(joint: np.ndarray, dim_names, n: int, seed: int = 42) -> pd.DataFrame:
    """Sample `n` agents. Each cell of `joint` is one combination of attributes;
    the chance of drawing it equals its share of the population."""
    rng = np.random.default_rng(seed)
    flat = joint.ravel() / joint.sum()
    cells = rng.choice(flat.size, size=n, p=flat)
    indices = np.unravel_index(cells, joint.shape)
    data = {"agent_id": np.arange(1, n + 1)}
    for name, idx in zip(dim_names, indices):
        cats = categories(name)
        data[name] = [cats[i] for i in idx]
    agents = pd.DataFrame(data)
    # Every agent stands for the same number of real people for now.
    # Weights change once rare groups are oversampled.
    agents["weight"] = 1.0
    return agents
