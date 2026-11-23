"""Give each agent a household: its type, size and total income.

Why: spending data is grouped by HOUSEHOLD income adjusted for household size
(equivalised income), but agents only carry personal income. A partnered
person with no income of their own may live in a high-income household.

Input:  TableBuilder export (.xlsx), 2021 Census, dataset "selected dwelling
        characteristics", filter STATE (EN) = New South Wales, built as
            rows    HIND   Total Household Income (weekly)
            columns HCFMD  Family Household Composition (Dwelling), 2-digit level
            wafers  NPRD   Number of Persons Usually Resident in Dwelling
Output: data/processed/households/nsw_households.csv  (tidy copy of the table)

Method. An agent's living arrangement fixes which household types they can be
in and how many people in each household hold that role:

    partnered             couple households, 2 partners each
    lone_parent           one parent households, 1 each
    child_at_home         couple-with-children households (size - 2 each)
                          and one parent households (size - 1 each)
    other_in_family_home  "other family" households, every member
    group_household       group households, every member
    lone_person           lone person households, 1 each

Within those, a household's income cannot be lower than the agent's own
income. The joint distribution of (agent's income, household) is the one that
matches both the agents' income distribution and the census household
distribution while respecting that rule (fitted with IPF). A lone person's
household income is their own income.

Approximations, all stated in the README:
- Households with two or more families (about 2% of households) are left out.
- "Partial income stated" and "not stated" households are dropped.
- The count of children per household includes children under 15, who are not
  agents; adult children are assumed to be spread across households in
  proportion to the number of children.
- Incomes are drawn evenly within each census band. The open top band
  ($8,000 or more) is treated as $8,000 to $12,000.
- Equivalised income uses the OECD-modified scale (1 for the first adult, 0.5
  for each further person aged 15+, 0.3 for each child under 15). The census
  table does not give children's ages, so each child counts as
  0.3 x (share of children under 15) + 0.5 x (share aged 15+), with the share
  taken from the census relationship table.
- Household income is gross (before tax). The spending survey ranks households
  by income after tax. Ranks are used, not dollar amounts, so the difference
  matters only where tax changes the order of households.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .ipf import Constraint, ipf
from .schema import categories

HOUSEHOLD_TYPES = {
    "One family household: Couple family with no children": "couple_no_children",
    "One family household: Couple family with children": "couple_with_children",
    "One family household: One parent family": "one_parent",
    "One family household: Other family": "other_family",
    "Lone person household": "lone_person",
    "Group household": "group",
}
SIZES = {"One person": 1, "Two persons": 2, "Three persons": 3, "Four persons": 4, "Five persons": 5,
         "Six persons": 6, "Seven persons": 7, "Eight or more persons": 8}
TOP_BAND_UPPER = 12000.0

# living arrangement -> {household type: people with that role, given household size}
ROLES = {
    "partnered": {"couple_no_children": lambda n: 2, "couple_with_children": lambda n: 2},
    "lone_parent": {"one_parent": lambda n: 1},
    "child_at_home": {"couple_with_children": lambda n: n - 2, "one_parent": lambda n: n - 1},
    "other_in_family_home": {"other_family": lambda n: n},
    "group_household": {"group": lambda n: n},
    "lone_person": {"lone_person": lambda n: 1},
}
# lowest and highest weekly dollars in each personal income band of the schema
INCOME_BOUNDS = {"nil_or_negative": (0, 1), "$1-499": (1, 500), "$500-999": (500, 1000),
                 "$1000-1499": (1000, 1500), "$1500-1999": (1500, 2000), "$2000+": (2000, np.inf)}


def _band(label: str):
    """Census household income label -> (low, high) weekly dollars, or None."""
    label = str(label).strip()
    if label in ("Negative income", "Nil income"):
        return (0.0, 1.0)
    if not label.startswith("$"):
        return None
    first = label.split(" (")[0].replace("$", "").replace(",", "")
    if "or more" in first:
        return (float(first.split()[0]), TOP_BAND_UPPER)
    low, high = first.split("-")
    return (float(low), float(high) + 1)


def read_household_table(path) -> pd.DataFrame:
    """Long form: household_type, size, income_low, income_high, dwellings."""
    rows = []
    for sheet, raw in pd.read_excel(path, sheet_name=None, header=None).items():
        size = SIZES.get(re.sub(r"^\d+\.\s*", "", sheet).strip())
        if size is None:
            continue
        header = raw.index[raw.apply(lambda r: "Lone person household" in r.values, axis=1)]
        if len(header) == 0:
            raise ValueError(f"Sheet '{sheet}': could not find the household composition header row")
        labels = raw.loc[header[0]]
        for i in range(header[0] + 1, len(raw)):
            band = _band(raw.iat[i, 1])
            if band is None:
                continue
            for col, label in labels.items():
                kind = HOUSEHOLD_TYPES.get(str(label).strip())
                if kind and raw.iat[i, col] > 0:
                    rows.append((kind, size, band[0], band[1], float(raw.iat[i, col])))
    if not rows:
        raise ValueError("No data found. Is this the HIND x HCFMD x NPRD export?")
    out = pd.DataFrame(rows, columns=["household_type", "size", "income_low", "income_high", "dwellings"])
    return out.groupby(["household_type", "size", "income_low", "income_high"], as_index=False)["dwellings"].sum()


def child_under15_share(rlhp_path) -> float:
    """Share of children living with parents who are under 15, from the census
    relationship table (TableBuilder RLHP x INCP export)."""
    raw = pd.read_excel(rlhp_path, sheet_name=0, header=None)
    labels = raw.iloc[:, 1].astype(str).str.strip()
    totals = pd.to_numeric(raw.iloc[:, -1], errors="coerce")
    under15 = totals[labels.str.contains("child under 15")].sum()
    older = totals[labels.str.startswith(("Natural or adopted dependent student", "Dependent student",
                                          "Non-dependent natural", "Non-dependent step child",
                                          "Non-dependent foster child"))].sum()
    return float(under15 / (under15 + older))


def equivalence_scale(household_type: pd.Series, size: pd.Series, under15: float) -> pd.Series:
    child = 0.3 * under15 + 0.5 * (1 - under15)
    children = np.select([household_type == "couple_with_children", household_type == "one_parent"],
                         [size - 2, size - 1], default=0)
    adults = size - children
    return 1 + 0.5 * (adults - 1) + child * children


def quintile_cutoffs(households: pd.DataFrame, under15: float, seed: int = 0, draws: int = 400_000) -> list:
    """Equivalised income at the 20th, 40th, 60th and 80th percentile of people
    (everyone in the household counts, children included), as the ABS does."""
    rng = np.random.default_rng(seed)
    weights = (households["dwellings"] * households["size"]).to_numpy()
    pick = rng.choice(len(households), size=draws, p=weights / weights.sum())
    h = households.iloc[pick]
    income = rng.uniform(h["income_low"], h["income_high"])
    scale = equivalence_scale(h["household_type"], h["size"], under15).to_numpy()
    return [float(q) for q in np.quantile(income / scale, [0.2, 0.4, 0.6, 0.8])]


def _household_probabilities(living: str, households: pd.DataFrame, income_share: np.ndarray):
    """P(household row | agent's income band) for one living arrangement.
    Returns (candidate household rows, matrix [income band x candidate], converged)."""
    parts = []
    for kind, people in ROLES[living].items():
        part = households[households["household_type"] == kind].copy()
        part["people"] = part["dwellings"] * part["size"].map(people)
        parts.append(part[part["people"] > 0])
    cand = pd.concat(parts).reset_index(drop=True)
    bands = categories("income")
    low = np.array([INCOME_BOUNDS[b][0] for b in bands])[:, None]
    high = np.array([INCOME_BOUNDS[b][1] for b in bands])[:, None]
    h_low, h_high = cand["income_low"].to_numpy()[None, :], cand["income_high"].to_numpy()[None, :]

    if living == "lone_person":
        # household income is the person's own income
        allowed = (h_low >= low) & (h_low < high) | ((low == 0) & (h_low == 0))
        allowed &= ~((low > 0) & (h_low == 0))
        p = allowed * cand["people"].to_numpy()[None, :]
        # an income band with no matching census households falls back to all of them
        p[p.sum(axis=1) == 0] = cand["people"].to_numpy()
        return cand, p / p.sum(axis=1, keepdims=True), True

    allowed = (h_high > low).astype(float)            # household income not below own income
    present = income_share > 0
    fit = ipf(["own", "household"], [int(present.sum()), len(cand)],
              [Constraint(("own",), income_share[present]),
               Constraint(("household",), cand["people"].to_numpy())],
              seed=allowed[present], tol=1e-7, max_iter=3000)
    p = np.zeros((len(bands), len(cand)))
    p[present] = fit.table
    empty = p.sum(axis=1) == 0
    p[empty] = (allowed * cand["people"].to_numpy()[None, :])[empty]
    return cand, p / p.sum(axis=1, keepdims=True), fit.converged


def assign_households(agents: pd.DataFrame, households: pd.DataFrame, living_income: pd.DataFrame,
                      under15: float, seed: int = 42) -> pd.DataFrame:
    """Add household_type, household_size, household_income_weekly,
    equivalised_income_weekly and income_quintile to the agents."""
    rng = np.random.default_rng(seed)
    out = agents.copy()
    bands = categories("income")
    kinds = np.empty(len(out), dtype=object)
    sizes = np.zeros(len(out), dtype=int)
    incomes = np.zeros(len(out))
    report = {}
    for living in categories("living"):
        if not (out["living"] == living).any():
            continue
        share = (living_income[living_income["living"] == living].set_index("income")["count"]
                 .reindex(bands).fillna(0).to_numpy())
        cand, p, converged = _household_probabilities(living, households, share / share.sum())
        report[living] = bool(converged)
        for b, band in enumerate(bands):
            idx = np.flatnonzero((out["living"] == living) & (out["income"] == band))
            if len(idx) == 0:
                continue
            pick = cand.iloc[rng.choice(len(cand), size=len(idx), p=p[b])]
            kinds[idx] = pick["household_type"].to_numpy()
            sizes[idx] = pick["size"].to_numpy()
            own_low = INCOME_BOUNDS[band][0]
            low = np.maximum(pick["income_low"].to_numpy(), own_low if living != "lone_person" else 0)
            incomes[idx] = rng.uniform(low, np.maximum(pick["income_high"].to_numpy(), low))
    out["household_type"], out["household_size"] = kinds, sizes
    out["household_income_weekly"] = incomes.round(0)
    scale = equivalence_scale(out["household_type"], out["household_size"], under15)
    out["equivalised_income_weekly"] = (incomes / scale).round(0)
    cuts = quintile_cutoffs(households, under15)
    names = np.array(["lowest", "second", "third", "fourth", "highest"])
    out["income_quintile"] = names[np.searchsorted(cuts, out["equivalised_income_weekly"], side="right")]
    out.attrs["converged"] = report
    out.attrs["quintile_cutoffs"] = cuts
    return out


def convert(path, rlhp_path, out_dir="data/processed/households") -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = read_household_table(path)
    table.to_csv(out_dir / "nsw_households.csv", index=False)
    under15 = child_under15_share(rlhp_path)
    (out_dir / "parameters.json").write_text(json.dumps(
        {"child_under15_share": round(under15, 4),
         "quintile_cutoffs_equivalised_weekly": [round(c) for c in quintile_cutoffs(table, under15)],
         "top_band_upper": TOP_BAND_UPPER}, indent=2))
    return table


def main():
    p = argparse.ArgumentParser(description="Convert the household income TableBuilder export")
    p.add_argument("--file", required=True, help="HIND x HCFMD x NPRD export (.xlsx)")
    p.add_argument("--rlhp", required=True, help="RLHP x INCP export (.xlsx), for children's ages")
    p.add_argument("--out", default="data/processed/households")
    a = p.parse_args()
    t = convert(a.file, a.rlhp, a.out)
    print(f"households: {len(t)} cells, {int(t['dwellings'].sum()):,} dwellings")
    print((Path(a.out) / "parameters.json").read_text())


if __name__ == "__main__":
    main()
