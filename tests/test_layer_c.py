import numpy as np
import pandas as pd
import pytest

from src.population.layer_c import (build_layer_c, calibrate, load_config,
                                     lognormal_mean_one)
from src.population.schema import categories

CFG = load_config("config/layer_c.yaml")


def _agents(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.choice(categories("age"), n),
        "location": rng.choice(categories("location"), n),
        "income_quintile": rng.choice(["lowest", "second", "third", "fourth", "highest"], n),
        "weight": 1.0,
    })
    for part in CFG["spending_variation"].values():
        for cat in part["categories"]:
            df[f"spend_{cat}_weekly"] = 20.0
    return df


def test_lognormal_has_mean_one():
    m = lognormal_mean_one(np.random.default_rng(1), 0.6, 400_000)
    assert m.mean() == pytest.approx(1.0, abs=0.01)


def test_calibrate_hits_target_and_caps_at_one():
    p = calibrate(np.array([1.0, 1.0, 10.0]), 0.6)
    assert p.mean() == pytest.approx(0.6)
    assert p.max() <= 1.0


def test_layer_c_targets_and_ranges():
    out = build_layer_c(_agents(), CFG)
    for name in CFG["traits"]:
        assert out[name].between(0, 1).all()
    low = out.loc[out.income_quintile == "lowest", "price_sensitivity"].mean()
    high = out.loc[out.income_quintile == "highest", "price_sensitivity"].mean()
    assert low == pytest.approx(CFG["traits"]["price_sensitivity"]["means"]["lowest"], abs=0.02)
    assert high == pytest.approx(CFG["traits"]["price_sensitivity"]["means"]["highest"], abs=0.02)
    assert out["uses_delivery_apps"].mean() == pytest.approx(CFG["delivery_apps"]["nsw_target"], abs=0.015)
    assert out["diet_vegetarian"].mean() == pytest.approx(CFG["vegetarian"]["target"], abs=0.01)
    assert out["spend_restaurants_weekly"].mean() == pytest.approx(20.0, rel=0.03)
    assert (out["word_of_mouth_reach"] <= CFG["word_of_mouth"]["max"]).all()


def test_novelty_and_loyalty_negatively_correlated():
    out = build_layer_c(_agents(), CFG)
    assert out["novelty_seeking"].corr(out["habit_loyalty"]) < -0.2


def test_layer_c_is_reproducible():
    a = build_layer_c(_agents(2000), CFG)
    b = build_layer_c(_agents(2000), CFG)
    assert a.equals(b)
