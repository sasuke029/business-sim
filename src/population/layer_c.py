"""Build Layer C: behaviour and personality, the final step of Phase 1.

Usage (from the project root, after Layer B):
    python -m src.population.layer_c
    python -m src.population.layer_c --config config/my_variant.yaml   # for sensitivity tests

Reads   data/processed/agents_layer_b.csv and config/layer_c.yaml
Writes  data/processed/agents.csv            the complete Phase 1 population
        outputs/validation/layer_c_checks.csv
        outputs/validation/layer_c_traits.png

All numbers come from the config file, which marks each one as data or as
an assumption. Nothing here should hold a hard-coded behavioural number.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from .schema import categories


def load_config(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def lognormal_mean_one(rng, sigma: float, n: int) -> np.ndarray:
    """Log-normal draws whose mean is exactly 1 in expectation."""
    return rng.lognormal(mean=-sigma ** 2 / 2, sigma=sigma, size=n)


def calibrate(relative: np.ndarray, target: float) -> np.ndarray:
    """Scale relative likelihoods so their average equals `target`, keeping
    every probability at or below 1."""
    scale = target / relative.mean()
    for _ in range(50):
        p = np.minimum(relative * scale, 1.0)
        if abs(p.mean() - target) < 1e-9:
            break
        scale *= target / p.mean()
    return np.minimum(relative * scale, 1.0)


def add_spending_variation(agents, cfg, rng):
    out = agents.copy()
    for name, part in cfg["spending_variation"].items():
        m = lognormal_mean_one(rng, part["sigma"], len(out))
        out[f"spend_multiplier_{name}"] = m.round(3)
        for cat in part["categories"]:
            col = f"spend_{cat}_weekly"
            out[col] = (out[col] * m).round(2)
    return out


def add_traits(agents, cfg, rng):
    out = agents.copy()
    names = list(cfg["traits"])
    corr = np.array(cfg["trait_correlation"], dtype=float)
    if corr.shape != (len(names), len(names)):
        raise ValueError("trait_correlation must be square with one row per trait")
    z = rng.multivariate_normal(np.zeros(len(names)), corr, size=len(out))
    u = stats.norm.cdf(z)
    for i, name in enumerate(names):
        t = cfg["traits"][name]
        missing = set(out[t["mean_by"]]) - set(t["means"])
        if missing:
            raise ValueError(f"{name}: no mean given for {sorted(missing)}")
        mean = out[t["mean_by"]].map(t["means"]).astype(float).to_numpy()
        k = float(t["concentration"])
        out[name] = stats.beta.ppf(u[:, i], mean * k, (1 - mean) * k).round(3)
    return out


def add_word_of_mouth(agents, cfg, rng):
    out = agents.copy()
    w = cfg["word_of_mouth"]
    base = rng.lognormal(np.log(w["median"]), w["sigma"], len(out))
    reach = base * out["age"].map(w["age_factor"]).to_numpy()
    out["word_of_mouth_reach"] = np.clip(np.round(reach), 0, w["max"]).astype(int)
    return out


def add_delivery_apps(agents, cfg, rng):
    out = agents.copy()
    d = cfg["delivery_apps"]
    relative = out["age"].map(d["by_age"]).to_numpy() * np.where(
        out["location"] == "rest_of_nsw", d["rest_of_nsw_relative"], 1.0)
    weights = out["weight"].to_numpy()
    p = calibrate(relative, d["nsw_target"]) if np.allclose(weights, 1) else relative
    out["uses_delivery_apps"] = rng.random(len(out)) < p
    return out


def add_diet(agents, cfg, rng):
    out = agents.copy()
    v = cfg["vegetarian"]
    relative = out["age"].map(v["age_factor"]).to_numpy() * out["location"].map(v["location_factor"]).to_numpy()
    out["diet_vegetarian"] = rng.random(len(out)) < calibrate(relative, v["target"])
    h = cfg["halal"]
    p_halal = out["location"].map(h["islam_share"]).to_numpy() * h["observance"]
    out["diet_halal"] = rng.random(len(out)) < p_halal
    return out


def build_layer_c(agents: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rng = np.random.default_rng(cfg["seed"])
    for step in (add_spending_variation, add_traits, add_word_of_mouth, add_delivery_apps, add_diet):
        agents = step(agents, cfg, rng)
    return agents


def checks(before: pd.DataFrame, after: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    for part in cfg["spending_variation"].values():
        for cat in part["categories"]:
            col = f"spend_{cat}_weekly"
            rows.append(("Average weekly spend kept after variation ($)", cat,
                         round(after[col].mean(), 2), round(before[col].mean(), 2),
                         "Should be close: multipliers average 1"))
    for name, t in cfg["traits"].items():
        for group, target in t["means"].items():
            sub = after[after[t["mean_by"]] == group]
            if len(sub):
                rows.append((f"Mean {name} by {t['mean_by']}", group,
                             round(sub[name].mean(), 3), target, f"n={len(sub)}"))
    d = cfg["delivery_apps"]
    rows.append(("Uses delivery apps (%)", "NSW", round(100 * after["uses_delivery_apps"].mean(), 1),
                 100 * d["nsw_target"], "Roy Morgan 2022"))
    rows.append(("Vegetarian (%)", "NSW", round(100 * after["diet_vegetarian"].mean(), 1),
                 100 * cfg["vegetarian"]["target"], "Roy Morgan 2018"))
    h = cfg["halal"]
    for loc in categories("location"):
        sub = after[after["location"] == loc]
        rows.append(("Looks for halal food (%)", loc, round(100 * sub["diet_halal"].mean(), 1),
                     round(100 * h["islam_share"][loc] * h["observance"], 1), f"Census G14 x assumption, n={len(sub)}"))
    w = after["word_of_mouth_reach"]
    rows.append(("Word of mouth reach", "median", float(w.median()), cfg["word_of_mouth"]["median"],
                 f"top 10% tell {int(w.quantile(0.9))}+ people"))
    return pd.DataFrame(rows, columns=["check", "group", "agents", "reference", "note"])


def plot_traits(agents: pd.DataFrame, cfg: dict, out_path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trait_names = list(cfg["traits"])
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    colors = ["#1F3A5F", "#E08E45", "#3C8D6E"]
    for ax, name, color in zip(axes[0], trait_names, colors):
        by = cfg["traits"][name]["mean_by"]
        order = list(cfg["traits"][name]["means"])
        data = [agents.loc[agents[by] == g, name] for g in order]
        ax.boxplot(data, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor=color, alpha=0.35, color=color), medianprops=dict(color=color))
        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels(order, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(f"{name.replace('_', ' ').capitalize()} by {by.replace('_', ' ')}", loc="left", fontweight="bold", fontsize=10)
    ax = axes[1][0]
    ax.hist(agents["spend_restaurants_weekly"], bins=40, color="#1F3A5F", alpha=0.8)
    ax.set_title("Restaurant spend per week ($)", loc="left", fontweight="bold", fontsize=10)
    ax = axes[1][1]
    ax.hist(agents["word_of_mouth_reach"], bins=range(0, 41), color="#E08E45", alpha=0.8)
    ax.set_title("Word of mouth reach (people per month)", loc="left", fontweight="bold", fontsize=10)
    ax = axes[1][2]
    order = categories("age")
    share = agents.groupby("age")["uses_delivery_apps"].mean().reindex(order) * 100
    ax.bar(range(len(order)), share, color="#3C8D6E", alpha=0.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("%")
    ax.set_title("Uses delivery apps, by age (%)", loc="left", fontweight="bold", fontsize=10)
    for a in axes.ravel():
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def build(config="config/layer_c.yaml", processed="data/processed", report="outputs/validation"):
    processed, report = Path(processed), Path(report)
    cfg = load_config(config)
    before = pd.read_csv(processed / "agents_layer_b.csv")
    after = build_layer_c(before, cfg)
    after.to_csv(processed / "agents.csv", index=False)
    table = checks(before, after, cfg)
    report.mkdir(parents=True, exist_ok=True)
    table.to_csv(report / "layer_c_checks.csv", index=False)
    plot_traits(after, cfg, report / "layer_c_traits.png")
    return after, table


def main():
    p = argparse.ArgumentParser(description="Build Layer C (behaviour and personality)")
    p.add_argument("--config", default="config/layer_c.yaml")
    p.add_argument("--processed", default="data/processed")
    p.add_argument("--report", default="outputs/validation")
    a = p.parse_args()
    agents, table = build(a.config, a.processed, a.report)
    print(f"Agents: {len(agents)} with {agents.shape[1]} columns written to {a.processed}/agents.csv")
    pd.set_option("display.width", 200)
    print(table.drop(columns="note").to_string(index=False))


if __name__ == "__main__":
    main()
