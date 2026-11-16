"""Load census tables saved in tidy form and turn them into IPF constraints.

Tidy form: one CSV per table, one row per cell. Columns are attribute names
from schema.py (or a grouped column such as `age_broad`) plus a `count` column.

    age,sex,count
    15-19,female,231000
    15-19,male,244000
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .ipf import Constraint
from .schema import CORE_ATTRIBUTES, GROUPED_COLUMNS, categories


def table_to_constraint(df: pd.DataFrame, name: str = "table") -> Constraint:
    if "count" not in df.columns:
        raise ValueError(f"{name}: missing 'count' column")
    if (df["count"] < 0).any():
        raise ValueError(f"{name}: negative counts")

    dims, col_cats, groups, labels = [], [], {}, {}
    for col in (c for c in df.columns if c != "count"):
        if col in GROUPED_COLUMNS:
            attr, mapping = GROUPED_COLUMNS[col]
            broad = list(mapping)
            fine_to_broad = {f: i for i, b in enumerate(broad) for f in mapping[b]}
            groups[attr] = [fine_to_broad[f] for f in categories(attr)]
            labels[attr] = broad
            dims.append(attr)
            col_cats.append(broad)
        else:
            dims.append(col)
            col_cats.append(categories(col))
        unknown = set(df[col]) - set(col_cats[-1])
        if unknown:
            raise ValueError(f"{name}: unknown {col} values {sorted(unknown)}")

    target = np.zeros(tuple(len(c) for c in col_cats))
    cols = [c for c in df.columns if c != "count"]
    for row in df.itertuples(index=False):
        values = dict(zip(df.columns, row))
        idx = tuple(cats.index(values[col]) for col, cats in zip(cols, col_cats))
        target[idx] += values["count"]
    if target.sum() <= 0:
        raise ValueError(f"{name}: table is empty")
    # Census tables are randomly perturbed and can cover slightly different
    # populations, so totals differ by a little. Proportions make them comparable.
    return Constraint(tuple(dims), target / target.sum(), groups, labels)


def load_constraints(folder) -> dict:
    """Read every CSV in `folder`. Returns {table_name: Constraint}."""
    folder = Path(folder)
    files = sorted(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV tables found in {folder}")
    constraints = {f.stem: table_to_constraint(pd.read_csv(f), f.name) for f in files}
    covered = {d for c in constraints.values() for d in c.dims if d not in c.groups}
    missing = set(CORE_ATTRIBUTES) - covered
    if missing:
        raise ValueError(f"No table covers these attributes in full detail: {sorted(missing)}")
    return constraints
