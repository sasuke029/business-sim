"""Build the Layer A synthetic population.

Usage (from the project root):
    python -m src.population.build --tables data/processed/tables --n 2000
"""
import argparse
from pathlib import Path

from .ipf import ipf
from .sample import sample_agents
from .schema import CORE_ATTRIBUTES
from .tables import load_constraints
from .validate import (compare_marginals, compare_tables, plot_marginals,
                       structural_zero_violations)


def build(tables_dir, n=2000, seed=42, out_dir="data/processed", report_dir="outputs/validation"):
    dim_names = list(CORE_ATTRIBUTES)
    dim_sizes = [len(v) for v in CORE_ATTRIBUTES.values()]

    constraints = load_constraints(tables_dir)
    result = ipf(dim_names, dim_sizes, list(constraints.values()))
    agents = sample_agents(result.table, dim_names, n=n, seed=seed)

    out_dir, report_dir = Path(out_dir), Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    agents.to_csv(out_dir / "agents_layer_a.csv", index=False)

    tables = compare_tables(agents, constraints)
    marginals = compare_marginals(agents, constraints)
    tables.to_csv(report_dir / "table_fit.csv", index=False)
    marginals.to_csv(report_dir / "marginal_fit.csv", index=False)
    # Never let invented test numbers be mistaken for census data on a chart.
    is_fake = "fake" in Path(tables_dir).name.lower()
    plot_marginals(marginals, report_dir / "marginals.png",
                   source_label="FAKE test tables (not census)" if is_fake else "Census")

    return {"agents": agents, "ipf": result, "tables": tables, "marginals": marginals,
            "impossible_agents": structural_zero_violations(agents, constraints)}


def main():
    p = argparse.ArgumentParser(description="Build the Layer A synthetic population")
    p.add_argument("--tables", required=True, help="Folder of tidy census CSV tables")
    p.add_argument("--n", type=int, default=2000, help="Number of agents")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/processed")
    p.add_argument("--report", default="outputs/validation")
    a = p.parse_args()

    r = build(a.tables, a.n, a.seed, a.out, a.report)
    fit = r["ipf"]
    print(f"IPF: converged={fit.converged} after {fit.iterations} iterations (max error {fit.max_error:.2e})")
    print(f"Agents: {len(r['agents'])} written to {a.out}/agents_layer_a.csv")
    print(f"Agents in combinations the census says are empty: {r['impossible_agents']}")
    print("\nFit against each census table (percentage points):")
    print(r["tables"].to_string(index=False))
    print(f"\nLargest single-attribute gap: {r['marginals']['gap_pct_points'].abs().max()} percentage points")
    print(f"Charts and tables saved in {a.report}/")


if __name__ == "__main__":
    main()
