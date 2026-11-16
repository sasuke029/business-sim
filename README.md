# business-sim

A virtual market of AI agents for stress-testing new business ideas.
The full plan is in `docs/Project_Plan.docx`.

**Status:** Phase 1 (synthetic population). Layer A is built on real 2021 Census
data for NSW: 2,000 agents with location, age, sex, income, work status and
living arrangement.

## Setup

```
python -m venv .venv
.venv\Scripts\activate          (Windows)
pip install -r requirements.txt
```

## Run the tests

```
python -m pytest
```

## Build the population

```
python -m src.population.census_gcp --datapack data/raw/2021_GCP_GCCSA_for_NSW_short-header.zip
python -m src.population.census_tablebuilder --file data/raw/tablebuilder_incp_lfsp_age5p_nsw.xlsx
python -m src.population.build --tables data/processed/tables --n 2000
```

The first two commands turn the ABS downloads into tidy tables in
`data/processed/tables/`. The third fits the population and writes:

- `data/processed/agents_layer_a.csv` - one row per agent
- `outputs/validation/marginals.png` - census vs synthetic, per attribute
- `outputs/validation/table_fit.csv` and `marginal_fit.csv` - the numbers behind the chart

## Data sources

Both from the Australian Bureau of Statistics, 2021 Census. Save downloads in
`data/raw/` (not committed to git).

1. **DataPack**: General Community Profile, Greater Capital City Statistical
   Areas, New South Wales.
   https://www.abs.gov.au/census/find-census-data/datapacks
2. **TableBuilder export** (free account needed): dataset "2021 Census -
   employment, income and education", filter STATE (UR) = New South Wales,
   rows INCP, columns LFSP, wafers AGE5P, downloaded as Excel.
   https://tablebuilder.abs.gov.au

| Table | Census source | What it gives |
|---|---|---|
| `location_age_sex` | G04 Age by sex | Population base, people aged 15+ |
| `location_age_sex_income` | G17 Total personal income (weekly) by age by sex | Income |
| `location_age_sex_work` | G46 Labour force status by age by sex | Work status |
| `location_age_sex_living` | G27 Relationship in household by age by sex | Living arrangement |
| `age_income_work` | TableBuilder INCP x LFSP x AGE5P | How income and work status go together at each age |

The cleaning choices (how "not stated" is handled, how tables are aligned) are
documented at the top of `src/population/census_gcp.py` and
`src/population/census_tablebuilder.py`.

## How Layer A works

1. Census tables are saved in tidy form: one CSV per table, attribute columns plus `count`.
2. `src/population/ipf.py` fits one joint distribution that agrees with every table
   (iterative proportional fitting).
3. `src/population/sample.py` draws agents from that distribution.
4. `src/population/validate.py` checks the agents against the census tables.

Attribute categories live in `src/population/schema.py`.

## Known limitations

- **Living arrangement is only linked to income and work through location, age
  and sex.** Within a group of the same location, age and sex, a person's
  living arrangement is treated as unrelated to their income and work status.
  A TableBuilder table crossing relationship in household with income would
  fix this the same way the income-by-work table did.
- **The income-by-work link is the same for both sexes and both locations.**
  The TableBuilder table is broken down by age only.
- **Location has two categories** (Greater Sydney, rest of NSW), not the four in
  the project plan.
- **Living arrangement** describes the person's place in the household
  (partnered, lone parent, living with parents, and so on). It does not say
  whether a couple has children.

## Fake test data

`data/sample_fake/` holds invented tables made by `scripts/make_fake_tables.py`.
They exist only so the tests and code can run without the census download.
Never use them for results.

## Layout

```
docs/               project plan
data/raw/           downloaded census files (not in git)
data/processed/     tidy census tables and agents
data/sample_fake/   invented tables for testing only
notebooks/          walkthroughs
scripts/            one-off scripts
src/population/     Phase 1 code
tests/              pytest tests
outputs/            validation charts and tables
```

## Effect of the income-by-work table

Share of full-time workers in each weekly income band, in the fitted population:

| | Nil or negative | $2000+ |
|---|---|---|
| Community Profile tables only | 6.9% | 20.1% |
| With the TableBuilder table | 0.4% | 31.9% |
| Census (TableBuilder, direct) | 0.4% | 32.2% |
