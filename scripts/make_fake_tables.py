"""Create FAKE census-style tables for testing the pipeline.

THESE NUMBERS ARE INVENTED. They are not census data and must never be used
for results. They exist so the code can be run and tested before the real
ABS tables are in place.

Usage: python scripts/make_fake_tables.py
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.population.schema import CORE_ATTRIBUTES, categories  # noqa: E402

rng = np.random.default_rng(0)
names = list(CORE_ATTRIBUTES)
sizes = [len(v) for v in CORE_ATTRIBUTES.values()]

# Start from random noise, then add a few crude, made-up patterns so the tables
# are not uniform and include some empty cells.
joint = rng.uniform(0.5, 1.5, size=sizes)
age, inc, work, liv = (names.index(k) for k in ("age", "income", "work", "living"))


def scale(axis_a, i, axis_b, j, factor):
    idx = [slice(None)] * joint.ndim
    idx[axis_a], idx[axis_b] = i, j
    joint[tuple(idx)] *= factor


scale(age, 0, inc, 5, 0.0)      # no 15-19 year olds in the top income band
scale(age, 0, inc, 4, 0.0)
scale(age, 0, liv, 2, 8.0)      # most 15-19 year olds live with parents
scale(age, 8, work, 0, 0.0)     # nobody 85+ working full time
scale(age, 8, work, 3, 9.0)
scale(age, 7, work, 3, 6.0)
scale(age, 3, liv, 0, 3.0)
joint /= joint.sum()

TOTAL = 6_500_000
TABLES = {
    "location_age_sex": ("location", "age", "sex"),
    "location_age_sex_income": ("location", "age", "sex", "income"),
    "location_age_sex_work": ("location", "age", "sex", "work"),
    "location_age_sex_living": ("location", "age", "sex", "living"),
}

out = Path(__file__).resolve().parents[1] / "data" / "sample_fake"
out.mkdir(parents=True, exist_ok=True)
for name, dims in TABLES.items():
    axes = tuple(names.index(d) for d in dims)
    other = tuple(i for i in range(joint.ndim) if i not in axes)
    marg = np.transpose(joint.sum(axis=other), np.argsort(np.argsort(axes)))
    rows = []
    for combo in itertools.product(*[range(len(categories(d))) for d in dims]):
        rows.append([categories(d)[i] for d, i in zip(dims, combo)] + [int(round(marg[combo] * TOTAL))])
    pd.DataFrame(rows, columns=list(dims) + ["count"]).to_csv(out / f"{name}.csv", index=False)
print(f"Wrote {len(TABLES)} FAKE tables to {out}")
