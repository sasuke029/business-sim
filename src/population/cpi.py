"""Price change since the 2015-16 spending survey, from the ABS Consumer Price Index.

Input:  CPI Table 18 (quarterly index numbers by group, sub-group and
        expenditure class, weighted average of eight capital cities), .xlsx
Output: data/processed/spending/cpi_factors.csv

Each factor = latest quarterly index / average index over the four quarters
of 2015-16. Multiplying a 2015-16 amount by it gives the cost of the same
purchases at current prices. It does NOT capture changes in how much people
buy (for example the growth of delivery apps since 2016).
"""
import argparse
from pathlib import Path

import pandas as pd

# our spending category -> CPI series
SERIES = {
    "restaurants": "Restaurant meals",
    "takeaway": "Take away and fast foods",
    "takeaway_coffee": "Take away and fast foods",
    "groceries": "Food and non-alcoholic beverages",
    "alcohol": "Alcoholic beverages",
    "clothing_footwear": "Clothing and footwear",
    "household_goods": "Furnishings, household equipment and services",
    "personal_care": "Personal care products",
    "recreation": "Recreation and culture",
}
BASE_QUARTERS = pd.to_datetime(["2015-09-01", "2015-12-01", "2016-03-01", "2016-06-01"])


def read_factors(path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="Data1", header=None)
    names = raw.iloc[0].astype(str)
    data = raw.iloc[10:].copy()
    dates = pd.to_datetime(data.iloc[:, 0])
    rows = []
    for category, series in SERIES.items():
        col = names[names == f"Index Numbers ;  {series} ;  Australia ;"].index
        if len(col) != 1:
            raise ValueError(f"CPI series '{series}' not found exactly once")
        values = pd.to_numeric(data[col[0]], errors="coerce")
        base = values[dates.isin(BASE_QUARTERS).to_numpy()].mean()
        latest = values.dropna().iloc[-1]
        latest_date = dates[values.dropna().index[-1]]
        rows.append({"category": category, "cpi_series": series, "index_2015_16": round(base, 2),
                     "index_latest": latest, "latest_quarter": latest_date.strftime("%b %Y"),
                     "factor": round(latest / base, 4)})
    return pd.DataFrame(rows)


def convert(path, out_dir="data/processed/spending") -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = read_factors(path)
    table.to_csv(out_dir / "cpi_factors.csv", index=False)
    return table


def main():
    p = argparse.ArgumentParser(description="Price change factors from CPI Table 18")
    p.add_argument("--file", required=True)
    p.add_argument("--out", default="data/processed/spending")
    a = p.parse_args()
    print(convert(a.file, a.out).to_string(index=False))


if __name__ == "__main__":
    main()
