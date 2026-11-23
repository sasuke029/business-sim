"""Add links between attributes from ABS TableBuilder exports.

The General Community Profile breaks every table down by location, age and
sex and nothing else, so on its own the pipeline treats income, work status
and living arrangement as unrelated within each of those groups. The tables
read here cross those attributes directly.

Inputs: TableBuilder exports (.xlsx), 2021 Census, dataset "employment, income
and education", filter STATE (UR) = New South Wales. Two layouts are recognised:

    1. INCP x LFSP x AGE5P  (rows income, columns labour force status, wafers age)
       -> data/processed/tables/age_income_work.csv
    2. RLHP x INCP          (rows relationship in household, columns income)
       -> data/processed/tables/living_income.csv

Usage (run after census_gcp, from the project root):
    python -m src.population.census_tablebuilder \
        --file data/raw/tablebuilder_incp_lfsp_age5p_nsw.xlsx \
        --file data/raw/tablebuilder_rlhp_incp_nsw.xlsx

Choices made here:
- "Not stated" and "Not applicable" answers are dropped.
- "Employed, away from work" is split between full-time and part-time in the
  same ratio as people of the same age and income whose hours are known.
- Relationship in household is grouped the same way as in census_gcp.
  Children under 15, visitors and unclassifiable relationships are left out.
- These tables and the Community Profile tables handle missing answers
  differently, so their totals disagree slightly. Each table is adjusted
  (raked) so its totals match the Community Profile tables exactly. That keeps
  what the table is for, the strength of the link between its attributes, and
  lets the totals come from one place.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .census_gcp import _income_band
from .ipf import Constraint, ipf
from .schema import categories

WORK = {
    "Employed, worked full-time": "full_time",
    "Employed, worked part-time": "part_time",
    "Employed, away from work": "away",
    "Unemployed, looking for full-time work": "unemployed",
    "Unemployed, looking for part-time work": "unemployed",
    "Not in the labour force": "not_in_labour_force",
}


LIVING_PREFIXES = [
    ("Husband, Wife or Partner", "partnered"),
    ("Lone parent", "lone_parent"),
    ("Natural or adopted dependent student", "child_at_home"),
    ("Dependent student", "child_at_home"),
    ("Non-dependent natural", "child_at_home"),
    ("Non-dependent step child", "child_at_home"),
    ("Non-dependent foster child", "child_at_home"),
    ("Brother/sister", "other_in_family_home"),
    ("Father/mother", "other_in_family_home"),
    ("Non-dependent grandchild", "other_in_family_home"),
    ("Grandfather/grandmother", "other_in_family_home"),
    ("Cousin", "other_in_family_home"),
    ("Uncle/aunt", "other_in_family_home"),
    ("Nephew/niece", "other_in_family_home"),
    ("Other related individual", "other_in_family_home"),
    ("Unrelated individual living in family household", "other_in_family_home"),
    ("Group household member", "group_household"),
    ("Lone person", "lone_person"),
]


def _living(label: str):
    label = str(label).strip()
    return next((cat for prefix, cat in LIVING_PREFIXES if label.startswith(prefix)), None)


def _age_band(sheet_name: str):
    m = re.match(r"^\d+\.\s*(\d+)(?:-\d+)? years", sheet_name)
    if not m:
        return None                                   # "Total" and helper sheets
    low = int(m.group(1))
    if low < 15:
        return None
    if low < 25:
        return "15-19" if low < 20 else "20-24"
    if low >= 85:
        return "85+"
    start = (low - 5) // 10 * 10 + 5                  # 25-29 and 30-34 -> 25-34
    return f"{start}-{start + 9}"


def _income_label(label: str):
    label = str(label).strip()
    if label in ("Negative income", "Nil income"):
        return "nil_or_negative"
    if not label.startswith("$"):
        return None                                   # Not stated, Not applicable, Total
    return _income_band(label.split(" ")[0].replace("$", "").replace(",", "").replace("-", "_"))


def read_export(path) -> pd.DataFrame:
    """Read the TableBuilder workbook into long form: age, income, work, count."""
    rows = []
    for sheet, raw in pd.read_excel(path, sheet_name=None, header=None).items():
        age = _age_band(sheet)
        if age is None:
            continue
        header = raw.index[raw.iloc[:, 0].astype(str).str.startswith("LFSP")]
        if len(header) == 0:
            raise ValueError(f"Sheet '{sheet}': could not find the LFSP header row")
        labels = raw.loc[header[0]]
        for i in range(header[0] + 1, len(raw)):
            income = _income_label(raw.iat[i, 1])
            if income is None:
                continue
            for col, label in labels.items():
                work = WORK.get(str(label).strip())
                if work:
                    rows.append((age, income, work, float(raw.iat[i, col])))
    if not rows:
        raise ValueError("No data found. Is this the INCP x LFSP x AGE5P export?")
    long = pd.DataFrame(rows, columns=["age", "income", "work", "count"])
    wide = long.pivot_table(index=["age", "income"], columns="work", values="count", aggfunc="sum").fillna(0)
    known = wide["full_time"] + wide["part_time"]
    share_ft = (wide["full_time"] / known).where(known > 0, 0.5)
    wide["full_time"] += wide["away"] * share_ft
    wide["part_time"] += wide["away"] * (1 - share_ft)
    out = wide.drop(columns="away").stack().rename("count").reset_index()
    return out


def read_living_income(path) -> pd.DataFrame:
    """Read the RLHP x INCP workbook into long form: living, income, count."""
    raw = pd.read_excel(path, sheet_name=0, header=None)
    header = raw.index[raw.iloc[:, 0].astype(str).str.startswith("INCP")]
    if len(header) == 0:
        raise ValueError("Expected income (INCP) across the columns and relationship (RLHP) down the rows")
    labels = raw.loc[header[0]]
    rows, seen = [], set()
    for i in range(header[0] + 1, len(raw)):
        living = _living(raw.iat[i, 1])
        if living is None:
            continue
        seen.add(living)
        for col, label in labels.items():
            income = _income_label(label) if col >= 2 else None
            if income:
                rows.append((living, income, float(raw.iat[i, col])))
    missing = set(categories("living")) - seen
    if missing:
        raise ValueError(f"The table has no rows for: {sorted(missing)}. "
                         "Select every RLHP category before downloading.")
    out = pd.DataFrame(rows, columns=["living", "income", "count"])
    return out.groupby(["living", "income"], as_index=False)["count"].sum()


def _rake(cross: pd.DataFrame, dims: list, margins: list, total: float) -> pd.DataFrame:
    """Adjust `cross` so it matches each (dims, table) pair in `margins`,
    keeping the strength of association in `cross`."""
    seed = _array(cross, dims)
    fit = ipf(dims, list(seed.shape),
              [Constraint(tuple(m_dims), _array(m_table, list(m_dims))) for m_dims, m_table in margins],
              seed=seed, tol=1e-6, max_iter=2000)
    if not fit.converged:
        raise RuntimeError("Could not reconcile the TableBuilder table with the Community Profile tables")
    idx = pd.MultiIndex.from_product([categories(d) for d in dims], names=dims)
    out = pd.Series((fit.table * total).round().astype(int).ravel(), index=idx, name="count")
    return out.reset_index()


def rake_living_income(cross: pd.DataFrame, tables_dir) -> pd.DataFrame:
    tables_dir = Path(tables_dir)
    income = pd.read_csv(tables_dir / "location_age_sex_income.csv")
    living = pd.read_csv(tables_dir / "location_age_sex_living.csv")
    return _rake(cross, ["living", "income"],
                 [(("living",), living), (("income",), income)], income["count"].sum())


def _array(df: pd.DataFrame, dims) -> np.ndarray:
    arr = np.zeros([len(categories(d)) for d in dims])
    for row in df.itertuples(index=False):
        arr[tuple(categories(d).index(getattr(row, d)) for d in dims)] += row.count
    return arr


def rake_to_profile(cross: pd.DataFrame, tables_dir) -> pd.DataFrame:
    """Adjust the income x work table so its age x income and age x work totals
    match the Community Profile tables already in `tables_dir`."""
    tables_dir = Path(tables_dir)
    income = pd.read_csv(tables_dir / "location_age_sex_income.csv")
    work = pd.read_csv(tables_dir / "location_age_sex_work.csv")
    return _rake(cross, ["age", "income", "work"],
                 [(("age", "income"), income), (("age", "work"), work)], income["count"].sum())


def detect_layout(path) -> str:
    title = " ".join(str(v) for v in pd.read_excel(path, sheet_name=0, header=None, nrows=4).values.ravel())
    if "RLHP" in title and "INCP" in title:
        return "living_income"
    if "AGE5P" in title and "LFSP" in title and "INCP" in title:
        return "age_income_work"
    raise ValueError(f"{path}: not a table layout this module knows. See the module docstring.")


def convert(path, tables_dir="data/processed/tables"):
    """Convert one export. Returns (table name, table)."""
    layout = detect_layout(path)
    if layout == "living_income":
        table = rake_living_income(read_living_income(path), tables_dir)
    else:
        table = rake_to_profile(read_export(path), tables_dir)
    table.to_csv(Path(tables_dir) / f"{layout}.csv", index=False)
    return layout, table


def main():
    p = argparse.ArgumentParser(description="Add linking tables from TableBuilder exports")
    p.add_argument("--file", required=True, action="append", help="TableBuilder .xlsx export (repeatable)")
    p.add_argument("--tables", default="data/processed/tables")
    a = p.parse_args()
    for f in a.file:
        name, t = convert(f, a.tables)
        print(f"{name}: {len(t)} cells, {int(t['count'].sum()):,} people")


if __name__ == "__main__":
    main()
