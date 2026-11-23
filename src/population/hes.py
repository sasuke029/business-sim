"""Spending by income level, from the ABS Household Expenditure Survey 2015-16.

Input:  "All data cubes" zip from the ABS Household Expenditure Survey page.
Output: data/processed/spending/hes_spend_by_income_quintile.csv

Usage (from the project root):
    python -m src.population.hes --zip data/raw/hes_2015-16_all_data_cubes.zip

The survey reports average weekly spending per HOUSEHOLD, for five groups of
households ranked by equivalised disposable income (income adjusted for
household size, so it measures living standard per person).

Choices made here:
- Spending is converted to "per adult" by dividing by the average number of
  adults (18+) in households of that income group. Children's spending is
  therefore carried by the adults in their household.
- Amounts are in 2015-16 dollars and Australia-wide. Scaling to current prices
  is a separate step.
- Groceries = all food and non-alcoholic beverages minus meals out and fast food.
"""
import argparse
import io
import zipfile
from pathlib import Path

import pandas as pd

QUINTILES = ["lowest", "second", "third", "fourth", "highest"]
WORKBOOK = "65300do003_201516.xls"        # equivalised disposable household income quintiles

# our category -> (how to find the row, text to match)
BROAD = {"alcohol": "Alcoholic beverages", "clothing_footwear": "Clothing and footwear",
         "household_goods": "Household furnishings and equipment", "recreation": "Recreation",
         "personal_care": "Personal care", "_food_total": "Food and non-alcoholic beverages"}
DETAILED = {
    "restaurants": "Meals in restaurants, hotels, clubs and related",
    "takeaway": "Fast food and takeaway exc. coffee (not frozen)",
    "takeaway_coffee": "Takeaway coffee",
    "_meals_out_total": "Meals out and fast foods",
}


def _first_text(row):
    return next((str(v).strip() for v in row if isinstance(v, str)), "")


def _quintile_columns(sheet: pd.DataFrame) -> list:
    header = sheet.index[sheet.apply(lambda r: "Lowest" in r.values and "Highest" in r.values, axis=1)][0]
    names = sheet.loc[header]
    return [names[names == q.capitalize()].index[0] for q in QUINTILES]


def read_spending(z: zipfile.ZipFile) -> pd.DataFrame:
    name = next(n for n in z.namelist() if n.lower().endswith(WORKBOOK))
    book = io.BytesIO(z.read(name))
    items = pd.read_excel(book, sheet_name="Table 3.3A", header=None)
    cols = _quintile_columns(items)
    codes = items[0].astype(str).str.strip()
    labels = items.iloc[:, 1:5].apply(_first_text, axis=1)

    household = {}
    for cat, text in BROAD.items():
        # broad groups are the rows that carry a two-digit code in the first column
        match = items.loc[codes.str.fullmatch(r"\d\d") & (labels == text), cols]
        if len(match) != 1:
            raise ValueError(f"Expected one broad group called '{text}', found {len(match)}")
        household[cat] = match.iloc[0].astype(float).values
    for cat, text in DETAILED.items():
        match = items.loc[labels == text.strip(), cols]
        if match.empty:
            raise ValueError(f"Could not find '{text}' in Table 3.3A")
        household[cat] = match.iloc[0].astype(float).values
    household["groceries"] = household.pop("_food_total") - household.pop("_meals_out_total")

    chars = pd.read_excel(book, sheet_name="Table 3.2", header=None)
    ccols = _quintile_columns(chars)
    clabels = chars.apply(_first_text, axis=1)
    estimates = chars.loc[: clabels[clabels.str.startswith("RELATIVE STANDARD ERROR")].index.min()] \
        if clabels.str.startswith("RELATIVE STANDARD ERROR").any() else chars
    def row(text):
        found = estimates.loc[clabels.loc[estimates.index] == text, ccols].dropna(how="all")
        return found.iloc[0].astype(float).values
    # persons per household from the population estimates (more precise than
    # the published averages, which are rounded to one decimal place)
    adults = row("Persons") / row("Households") - row("Under 18 years")
    mean_disposable = row("Mean disposable income")

    out = []
    for cat, values in household.items():
        for q, hh, n, inc in zip(QUINTILES, values, adults, mean_disposable):
            out.append({"category": cat, "income_quintile": q,
                        "household_weekly": round(hh, 2), "adults_per_household": round(n, 2),
                        "per_adult_weekly": round(hh / n, 2),
                        "mean_household_disposable_income": inc, "dollars": "2015-16"})
    return pd.DataFrame(out)


def convert(zip_path, out_dir="data/processed/spending") -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        table = read_spending(z)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "hes_spend_by_income_quintile.csv", index=False)
    return table


def main():
    p = argparse.ArgumentParser(description="Extract spending by income quintile from the HES data cubes")
    p.add_argument("--zip", required=True)
    p.add_argument("--out", default="data/processed/spending")
    a = p.parse_args()
    t = convert(a.zip, a.out)
    print(t.pivot(index="category", columns="income_quintile", values="per_adult_weekly")[QUINTILES].to_string())


if __name__ == "__main__":
    main()
