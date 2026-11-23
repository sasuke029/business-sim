"""Build Layer B: each agent's household and weekly spending.

Usage (from the project root, after Layer A and the converters have run):
    python -m src.population.layer_b

Reads   data/processed/agents_layer_a.csv
        data/processed/tables/living_income.csv
        data/processed/households/nsw_households.csv and parameters.json
        data/processed/spending/hes_spend_by_income_quintile.csv and cpi_factors.csv
Writes  data/processed/agents_layer_b.csv
        outputs/validation/layer_b_checks.csv

Spending columns are the AVERAGE weekly spend per adult for households at the
agent's income level, in current prices. Everyone in the same income quintile
gets the same figure; person-to-person variation is added in Layer C.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from .households import ROLES, assign_households

QUINTILES = ["lowest", "second", "third", "fourth", "highest"]


def add_spending(agents: pd.DataFrame, hes: pd.DataFrame, cpi: pd.DataFrame) -> pd.DataFrame:
    out = agents.copy()
    factor = cpi.set_index("category")["factor"]
    for category, part in hes.groupby("category"):
        per_adult = part.set_index("income_quintile")["per_adult_weekly"] * factor[category]
        out[f"spend_{category}_weekly"] = out["income_quintile"].map(per_adult).round(2)
    return out


def checks(agents: pd.DataFrame, households: pd.DataFrame, hes: pd.DataFrame, cpi: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for q, share in (agents["income_quintile"].value_counts(normalize=True).reindex(QUINTILES) * 100).items():
        rows.append(("Share of agents in income quintile", q, round(share, 1), 20.0,
                     "Census cut-offs put 20% of all people in each; agents are aged 15+ only"))
    # household income: agents vs the census households they could belong to
    for living, kinds in ROLES.items():
        parts = []
        for kind, role in kinds.items():
            h = households[households["household_type"] == kind]
            parts.append(h.assign(people=h["dwellings"] * h["size"].map(role)))
        census = pd.concat(parts)
        mine = agents[agents["living"] == living]
        for label, c_mask, a_mask in [
            ("household income under $1,000", census["income_high"] <= 1000, mine["household_income_weekly"] < 1000),
            ("household income $3,000 or more", census["income_low"] >= 3000, mine["household_income_weekly"] >= 3000),
        ]:
            rows.append((f"Share with {label} (%)", living, round(100 * a_mask.mean(), 1),
                         round(100 * census.loc[c_mask, "people"].sum() / census["people"].sum(), 1),
                         "Census: people in the household types this living arrangement can be in"))
    # spending: agents' average vs the survey's all-household average per adult
    factor = cpi.set_index("category")["factor"]
    for category, part in hes.groupby("category"):
        survey = (part["household_weekly"].mean() / part["adults_per_household"].mean()) * factor[category]
        rows.append(("Average weekly spend per adult ($, current prices)", category,
                     round(agents[f"spend_{category}_weekly"].mean(), 2), round(survey, 2),
                     "Survey: Australia-wide average across the five quintiles"))
    return pd.DataFrame(rows, columns=["check", "group", "agents", "reference", "note"])


def build(processed="data/processed", report="outputs/validation", seed=42):
    processed, report = Path(processed), Path(report)
    agents = pd.read_csv(processed / "agents_layer_a.csv")
    households = pd.read_csv(processed / "households" / "nsw_households.csv")
    params = json.loads((processed / "households" / "parameters.json").read_text())
    living_income = pd.read_csv(processed / "tables" / "living_income.csv")
    hes = pd.read_csv(processed / "spending" / "hes_spend_by_income_quintile.csv")
    cpi = pd.read_csv(processed / "spending" / "cpi_factors.csv")

    with_households = assign_households(agents, households, living_income,
                                        params["child_under15_share"], seed=seed)
    converged = with_households.attrs["converged"]
    result = add_spending(with_households, hes, cpi)
    result.to_csv(processed / "agents_layer_b.csv", index=False)
    table = checks(result, households, hes, cpi)
    report.mkdir(parents=True, exist_ok=True)
    table.to_csv(report / "layer_b_checks.csv", index=False)
    return result, table, converged


def main():
    p = argparse.ArgumentParser(description="Build Layer B (households and spending)")
    p.add_argument("--processed", default="data/processed")
    p.add_argument("--report", default="outputs/validation")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    agents, table, converged = build(a.processed, a.report, a.seed)
    print(f"Agents: {len(agents)} written to {a.processed}/agents_layer_b.csv")
    not_ok = [k for k, v in converged.items() if not v]
    print("Household fit converged for every living arrangement" if not not_ok
          else f"Household fit did NOT converge for: {not_ok}")
    print(table.drop(columns="note").to_string(index=False))


if __name__ == "__main__":
    main()
