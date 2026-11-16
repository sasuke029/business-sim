"""Compare the synthetic population with the census tables it was built from."""
from pathlib import Path

import numpy as np
import pandas as pd

from .ipf import coarsen
from .schema import CORE_ATTRIBUTES, categories


def _agent_table(agents: pd.DataFrame, dims) -> np.ndarray:
    shape = tuple(len(categories(d)) for d in dims)
    out = np.zeros(shape)
    grouped = agents.groupby(list(dims), observed=True)["weight"].sum()
    for key, w in grouped.items():
        key = key if isinstance(key, tuple) else (key,)
        out[tuple(categories(d).index(k) for d, k in zip(dims, key))] = w
    return out / out.sum()


def _agent_table_like(agents: pd.DataFrame, c) -> np.ndarray:
    """Agents tabulated the same way as constraint `c`, including broad bands."""
    return coarsen(_agent_table(agents, c.dims), [c.groups.get(d) for d in c.dims])


def compare_tables(agents: pd.DataFrame, constraints: dict) -> pd.DataFrame:
    """One row per input table: how far the agents are from the census."""
    rows = []
    for name, c in constraints.items():
        gap = np.abs(_agent_table_like(agents, c) - c.target)
        rows.append({
            "table": name,
            "dimensions": " x ".join(c.dims),
            "cells": gap.size,
            "max_gap_pct_points": round(100 * gap.max(), 2),
            "mean_gap_pct_points": round(100 * gap.mean(), 3),
        })
    return pd.DataFrame(rows)


def compare_marginals(agents: pd.DataFrame, constraints: dict) -> pd.DataFrame:
    """One row per category of each attribute: census share vs synthetic share."""
    rows = []
    for attr in CORE_ATTRIBUTES:
        source = next(c for c in constraints.values() if attr in c.dims and attr not in c.groups)
        axis = source.dims.index(attr)
        other = tuple(i for i in range(source.target.ndim) if i != axis)
        census = source.target.sum(axis=other)
        synthetic = _agent_table(agents, (attr,))
        for cat, c_share, s_share in zip(categories(attr), census, synthetic):
            rows.append({"attribute": attr, "category": cat,
                         "census_pct": round(100 * c_share, 2),
                         "synthetic_pct": round(100 * s_share, 2),
                         "gap_pct_points": round(100 * (s_share - c_share), 2)})
    return pd.DataFrame(rows)


def structural_zero_violations(agents: pd.DataFrame, constraints: dict) -> int:
    """Count agents that sit in a cell the census says is empty."""
    total = 0.0
    for c in constraints.values():
        synthetic = _agent_table_like(agents, c)
        total = max(total, synthetic[c.target == 0].sum())
    return int(round(total * len(agents)))


def plot_marginals(marginals: pd.DataFrame, out_path, source_label: str = "Census") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    attrs = list(marginals["attribute"].unique())
    cols = 2
    rows = (len(attrs) + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(13, 3.6 * rows))
    for ax, attr in zip(axes.ravel(), attrs):
        part = marginals[marginals["attribute"] == attr]
        x = np.arange(len(part))
        ax.bar(x - 0.2, part["census_pct"], 0.4, label=source_label, color="#1F3A5F")
        ax.bar(x + 0.2, part["synthetic_pct"], 0.4, label="Synthetic agents", color="#E08E45")
        ax.set_xticks(x)
        ax.set_xticklabels(part["category"], rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("% of population")
        ax.set_title(attr.capitalize(), loc="left", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.ravel()[len(attrs):]:
        ax.axis("off")
    axes.ravel()[0].legend(frameon=False)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
