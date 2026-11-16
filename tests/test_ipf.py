import numpy as np
import pandas as pd
import pytest

from src.population.ipf import Constraint, ipf
from src.population.sample import sample_agents
from src.population.schema import CORE_ATTRIBUTES
from src.population.tables import table_to_constraint


def test_ipf_matches_all_marginals():
    rng = np.random.default_rng(1)
    truth = rng.uniform(0.1, 1, size=(4, 3, 5))
    truth /= truth.sum()
    cons = [Constraint(("a", "b"), truth.sum(axis=2)),
            Constraint(("b", "c"), truth.sum(axis=0)),
            Constraint(("c", "a"), truth.sum(axis=1).T)]   # axes deliberately out of order
    r = ipf(["a", "b", "c"], [4, 3, 5], cons)
    assert r.converged
    assert np.allclose(r.table.sum(axis=2), truth.sum(axis=2), atol=1e-6)
    assert np.allclose(r.table.sum(axis=0), truth.sum(axis=0), atol=1e-6)
    assert np.allclose(r.table.sum(axis=1), truth.sum(axis=1), atol=1e-6)
    assert r.table.sum() == pytest.approx(1.0)


def test_ipf_keeps_empty_cells_empty():
    target = np.array([[0.0, 0.4], [0.3, 0.3]])
    r = ipf(["a", "b", "c"], [2, 2, 3], [Constraint(("a", "b"), target)])
    assert r.table[0, 0].sum() == 0


def test_ipf_rejects_wrong_shape():
    with pytest.raises(ValueError):
        ipf(["a", "b"], [2, 2], [Constraint(("a",), np.ones(3))])


def test_table_rejects_unknown_category():
    df = pd.DataFrame({"sex": ["female", "robot"], "count": [1, 1]})
    with pytest.raises(ValueError):
        table_to_constraint(df)


def test_sampling_is_reproducible_and_valid():
    names = list(CORE_ATTRIBUTES)
    sizes = [len(v) for v in CORE_ATTRIBUTES.values()]
    joint = np.ones(sizes) / np.prod(sizes)
    a = sample_agents(joint, names, n=500, seed=7)
    b = sample_agents(joint, names, n=500, seed=7)
    assert a.equals(b)
    assert len(a) == 500
    for name in names:
        assert set(a[name]) <= set(CORE_ATTRIBUTES[name])


def test_ipf_with_broad_bands():
    # joint has 3 fine categories on "a"; one table only knows a broad split {0,1} vs {2}
    rng = np.random.default_rng(3)
    truth = rng.uniform(0.1, 1, size=(3, 4))
    truth /= truth.sum()
    fine = Constraint(("a",), truth.sum(axis=1))
    broad_target = np.vstack([truth[0] + truth[1], truth[2]])
    broad = Constraint(("a", "b"), broad_target, groups={"a": [0, 0, 1]})
    r = ipf(["a", "b"], [3, 4], [fine, broad])
    assert r.converged
    assert np.allclose(r.table.sum(axis=1), truth.sum(axis=1), atol=1e-5)
    assert np.allclose(np.vstack([r.table[0] + r.table[1], r.table[2]]), broad_target, atol=1e-5)


def test_grouped_column_in_table():
    df = pd.DataFrame({"age_broad": ["15-24", "25-34"], "count": [10, 30]})
    c = table_to_constraint(df)
    assert c.dims == ("age",)
    assert c.groups["age"][:3] == [0, 0, 1]
    assert c.target[0] == pytest.approx(0.25)


def test_tablebuilder_label_parsing():
    from src.population.census_tablebuilder import _age_band, _income_label
    assert _age_band("3. 15-19 years") == "15-19"
    assert _age_band("5. 25-29 years") == "25-34"
    assert _age_band("6. 30-34 years") == "25-34"
    assert _age_band("16. 80-84 years") == "75-84"
    assert _age_band("20. 100 years and over") == "85+"
    assert _age_band("2. 10-14 years") is None
    assert _age_band("21. Total") is None
    assert _income_label("Nil income") == "nil_or_negative"
    assert _income_label("$400-$499 ($20,800-$25,999)") == "$1-499"
    assert _income_label("$1,000-$1,249 ($52,000-$64,999)") == "$1000-1499"
    assert _income_label("$3,500 or more ($182,000 or more)") == "$2000+"
    assert _income_label("Not stated") is None
