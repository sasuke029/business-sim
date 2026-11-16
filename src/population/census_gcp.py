"""Turn an ABS 2021 Census DataPack into the tidy tables the pipeline reads.

Input:  General Community Profile DataPack, Greater Capital City Statistical
        Areas, New South Wales (the zip as downloaded from the ABS).
Output: four CSV tables in data/processed/tables/

    location_age_sex.csv          from G04  Age by sex (the population base)
    location_age_sex_income.csv   from G17  Total personal income by age by sex
    location_age_sex_work.csv     from G46  Labour force status by age by sex
    location_age_sex_living.csv   from G27  Relationship in household by age by sex

Usage (from the project root):
    python -m src.population.census_gcp --datapack data/raw/2021_GCP_GCCSA_for_NSW_short-header.zip

Choices made here, all of which affect the population:
- People aged 15 and over only.
- "Not stated" answers are dropped, which spreads them across the stated
  answers in proportion.
- Employed people who were away from work, or whose hours were not stated, are
  split between full-time and part-time in the same ratio as people of the
  same age, sex and location whose hours are known.
- Visitors are left out of the household table.
- People with no usual address and offshore/migratory areas are left out.
- The income, work and household tables each count a slightly different set of
  people (for example the household table covers private dwellings only). Each
  is rescaled so its totals by location, age and sex match the age-by-sex
  table. Without this the tables contradict each other and no population can
  match all of them.
"""
import argparse
import io
import re
import zipfile
from pathlib import Path

import pandas as pd

LOCATIONS = {"1GSYD": "greater_sydney", "1RNSW": "rest_of_nsw"}
SEX = {"MALES": "male", "FEMALES": "female"}
AGE = {"15_19_years": "15-19", "20_24_years": "20-24", "25_34_years": "25-34",
       "35_44_years": "35-44", "45_54_years": "45-54", "55_64_years": "55-64",
       "65_74_years": "65-74", "75_84_years": "75-84", "85_years_and_over": "85+"}
AGE_BROAD = {**{k: v for k, v in AGE.items() if k not in ("15_19_years", "20_24_years")},
             "15_24_years": "15-24"}

LIVING = {
    "Partner_in_a_registered_marriage": "partnered",
    "Partner_in_de_facto_marriage": "partnered",
    "Lone_parent": "lone_parent",
    "Dependent_student_aged_15_24_years": "child_at_home",
    "Non_dependent_child": "child_at_home",
    "Other_related_individual": "other_in_family_home",
    "Unrelated_individual_living_in_family_household": "other_in_family_home",
    "Group_household_member": "group_household",
    "Lone_person": "lone_person",
}

_CELL = re.compile(r"^(MALES|FEMALES|PERSONS)_(.+?)_(?:Age_)?(\d+_\d+_years|\d+_years_and_over|Total)$")


def _income_band(label: str):
    if label.startswith("Negative_Nil"):
        return "nil_or_negative"
    m = re.match(r"(\d+)", label)
    if not m:
        return None                      # "Personal_income_not_stated", "Total"
    low = int(m.group(1))
    if low < 500:
        return "$1-499"
    if low < 1000:
        return "$500-999"
    if low < 1500:
        return "$1000-1499"
    if low < 2000:
        return "$1500-1999"
    return "$2000+"


def _read_table(z: zipfile.ZipFile, meta: pd.DataFrame, table: str) -> pd.DataFrame:
    """Return one census table in long form: location, sex, category, age, count."""
    frames = []
    for name in z.namelist():
        if re.search(rf"_{table}[A-Z]?_NSW_GCCSA\.csv$", name):
            wide = pd.read_csv(z.open(name))
            frames.append(wide.melt(id_vars=wide.columns[0], var_name="short", value_name="count")
                              .rename(columns={wide.columns[0]: "code"}))
    if not frames:
        raise FileNotFoundError(f"Table {table} not found in the DataPack")
    long = pd.concat(frames).merge(meta, on="short", how="left")
    # The ABS writes ".." in cells that cannot apply (e.g. a partner aged 0-14).
    long["count"] = pd.to_numeric(long["count"], errors="coerce").fillna(0)
    if long["long"].isna().any():
        raise ValueError(f"{table}: some columns are missing from the metadata file")
    parts = long["long"].str.extract(_CELL)
    if parts[0].isna().any():
        bad = long.loc[parts[0].isna(), "long"].unique()[:5]
        raise ValueError(f"{table}: could not read column names such as {list(bad)}")
    long["sex"], long["category"], long["age_raw"] = parts[0], parts[1], parts[2]
    long["location"] = long["code"].map(LOCATIONS)
    return long.dropna(subset=["location"])


def _load_meta(z: zipfile.ZipFile) -> pd.DataFrame:
    name = next(n for n in z.namelist() if "Metadata_2021_GCP_DataPack" in n)
    meta = pd.read_excel(io.BytesIO(z.read(name)), sheet_name="Cell Descriptors Information", header=10)
    meta = meta.iloc[:, [1, 2, 3]]
    meta.columns = ["short", "long", "file"]
    return meta.dropna(subset=["short"])


def _keep_people(long: pd.DataFrame, age_map: dict, age_col: str) -> pd.DataFrame:
    out = long[long["sex"].isin(SEX) & long["age_raw"].isin(age_map)].copy()
    out["sex"] = out["sex"].map(SEX)
    out[age_col] = out["age_raw"].map(age_map)
    return out


def income_table(z, meta) -> pd.DataFrame:
    t = _keep_people(_read_table(z, meta[meta["file"].str.startswith("G17")], "G17"), AGE, "age")
    t["income"] = t["category"].map(_income_band)
    t = t.dropna(subset=["income"])
    return t.groupby(["location", "age", "sex", "income"], as_index=False)["count"].sum()


def work_table(z, meta) -> pd.DataFrame:
    t = _keep_people(_read_table(z, meta[meta["file"].str.startswith("G46")], "G46"), AGE, "age")
    wide = t.pivot_table(index=["location", "age", "sex"], columns="category", values="count", aggfunc="sum")
    ft, pt = wide["Employed_worked_Full_time"], wide["Employed_worked_Part_time"]
    unknown_hours = wide["Employed_away_from_work"] + wide["Hours_worked_not_stated"]
    share_ft = (ft / (ft + pt)).fillna(0.5)
    out = pd.DataFrame({
        "full_time": ft + unknown_hours * share_ft,
        "part_time": pt + unknown_hours * (1 - share_ft),
        "unemployed": wide["Total_unemployed"],
        "not_in_labour_force": wide["Not_in_the_labour_force"],
    })
    out = out.stack().rename("count").reset_index().rename(columns={"level_3": "work"})
    out["count"] = out["count"].round().astype(int)
    return out


def living_table(z, meta) -> pd.DataFrame:
    t = _keep_people(_read_table(z, meta[meta["file"].str.startswith("G27")], "G27"), AGE_BROAD, "age_broad")
    t["living"] = t["category"].map(LIVING)
    t = t.dropna(subset=["living"])
    return t.groupby(["location", "age_broad", "sex", "living"], as_index=False)["count"].sum()


def base_table(z) -> pd.DataFrame:
    """Everyone aged 15+ by location, age band and sex, from G04 (Age by sex)."""
    wide = pd.concat([pd.read_csv(z.open(n)).set_index("GCCSA_CODE_2021")
                      for n in z.namelist() if re.search(r"_G04[A-Z]?_NSW_GCCSA\.csv$", n)], axis=1)
    rows = []
    for col in wide.columns:
        m = re.match(r"Age_yr_(\d+)_(?:\d+|yr_over)_([MF])$", col)
        if not m or int(m.group(1)) < 15:
            continue
        low = int(m.group(1))
        band = ("15-19" if low < 20 else "20-24" if low < 25 else "85+" if low >= 85
                else f"{low // 10 * 10 + 5}-{low // 10 * 10 + 14}" if low % 10 >= 5
                else f"{low // 10 * 10 - 5}-{low // 10 * 10 + 4}")
        for code, location in LOCATIONS.items():
            rows.append((location, band, "male" if m.group(2) == "M" else "female", wide.loc[code, col]))
    df = pd.DataFrame(rows, columns=["location", "age", "sex", "count"])
    return df.groupby(["location", "age", "sex"], as_index=False)["count"].sum()


def _align_to_base(table: pd.DataFrame, base: pd.DataFrame, age_col: str) -> pd.DataFrame:
    """Rescale `table` so its location x age x sex totals equal the base table."""
    base = base.copy()
    if age_col == "age_broad":
        base["age_broad"] = base["age"].replace({"15-19": "15-24", "20-24": "15-24"})
        base = base.groupby(["location", "age_broad", "sex"], as_index=False)["count"].sum()
    keys = ["location", age_col, "sex"]
    totals = table.groupby(keys)["count"].sum().rename("own_total").reset_index()
    out = table.merge(totals, on=keys).merge(base.rename(columns={"count": "base_total"}), on=keys)
    out["count"] = (out["count"] * out["base_total"] / out["own_total"]).round().astype(int)
    return out.drop(columns=["own_total", "base_total"])


def convert(datapack, out_dir="data/processed/tables") -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(datapack) as z:
        meta = _load_meta(z)
        base = base_table(z)
        tables = {
            "location_age_sex": base,
            "location_age_sex_income": _align_to_base(income_table(z, meta), base, "age"),
            "location_age_sex_work": _align_to_base(work_table(z, meta), base, "age"),
            "location_age_sex_living": _align_to_base(living_table(z, meta), base, "age_broad"),
        }
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    return tables


def main():
    p = argparse.ArgumentParser(description="Convert an ABS 2021 GCP DataPack into tidy tables")
    p.add_argument("--datapack", required=True, help="Path to the DataPack zip")
    p.add_argument("--out", default="data/processed/tables")
    a = p.parse_args()
    for name, df in convert(a.datapack, a.out).items():
        print(f"{name}: {len(df)} cells, {int(df['count'].sum()):,} people")


if __name__ == "__main__":
    main()
