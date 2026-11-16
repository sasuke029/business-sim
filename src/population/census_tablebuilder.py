"""Add the income-by-work-status link from an ABS TableBuilder export.

The General Community Profile has no table that crosses income with labour
force status, so on its own the pipeline treats them as unrelated within each
location, age and sex group. This module reads a TableBuilder table that does
cross them and writes it as one more tidy table for the pipeline.

Input:  TableBuilder export (.xlsx), 2021 Census, NSW filter, built as
            rows    INCP  Total Personal Income (weekly)
            columns LFSP  Labour Force Status
            wafers  AGE5P Age in Five Year Groups
Output: data/processed/tables/age_income_work.csv

Usage (run after census_gcp, from the project root):
    python -m src.population.census_tablebuilder --file data/raw/tablebuilder_incp_lfsp_age5p_nsw.xlsx

Choices made here:
- "Not stated" and "Not applicable" answers are dropped.
- "Employed, away from work" is split between full-time and part-time in the
  same ratio as people of the same age and income whose hours are known.
- This table and the Community Profile tables handle missing answers
  differently, so their income and work totals by age disagree slightly. The
  table is adjusted (raked) so its totals match the Community Profile tables
  exactly. That keeps what this table is for, the strength of the link between
  income and work at each age, and lets the totals come from one place.
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


def _array(df: pd.DataFrame, dims) -> np.ndarray:
    arr = np.zeros([len(categories(d)) for d in dims])
    for row in df.itertuples(index=False):
        arr[tuple(categories(d).index(getattr(row, d)) for d in dims)] += row.count
    return arr


def rake_to_profile(cross: pd.DataFrame, tables_dir) -> pd.DataFrame:
    """Adjust the cross table so its age x income and age x work totals match
    the Community Profile tables already in `tables_dir`."""
    tables_dir = Path(tables_dir)
    income = pd.read_csv(tables_dir / "location_age_sex_income.csv")
    work = pd.read_csv(tables_dir / "location_age_sex_work.csv")
    dims = ["age", "income", "work"]
    seed = _array(cross, dims)
    fit = ipf(dims, list(seed.shape),
              [Constraint(("age", "income"), _array(income, ["age", "income"])),
               Constraint(("age", "work"), _array(work, ["age", "work"]))],
              seed=seed, tol=1e-6, max_iter=2000)
    if not fit.converged:
        raise RuntimeError("Could not reconcile the TableBuilder table with the Community Profile tables")
    total = income["count"].sum()
    rows = [(a, i, w, int(round(fit.table[x, y, z] * total)))
            for x, a in enumerate(categories("age"))
            for y, i in enumerate(categories("income"))
            for z, w in enumerate(categories("work"))]
    return pd.DataFrame(rows, columns=dims + ["count"])


def convert(path, tables_dir="data/processed/tables") -> pd.DataFrame:
    table = rake_to_profile(read_export(path), tables_dir)
    table.to_csv(Path(tables_dir) / "age_income_work.csv", index=False)
    return table


def main():
    p = argparse.ArgumentParser(description="Add the income x work table from a TableBuilder export")
    p.add_argument("--file", required=True, help="TableBuilder .xlsx export")
    p.add_argument("--tables", default="data/processed/tables")
    a = p.parse_args()
    t = convert(a.file, a.tables)
    print(f"age_income_work: {len(t)} cells, {int(t['count'].sum()):,} people")


if __name__ == "__main__":
    main()
