# business-sim

A virtual market of AI agents for stress-testing new business ideas.
The full plan is in `docs/Project_Plan.docx`.

**Status:** Phase 1 (synthetic population) complete. 2,000 NSW agents in
`data/processed/agents.csv`, each with demographics (Layer A, census), a
household and weekly spending (Layer B, census + spending survey + CPI), and
behaviour and personality (Layer C, survey data plus stated assumptions).
Phase 2 (persona cards) is next.

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

Layer A (demographics):

```
python -m src.population.census_gcp --datapack data/raw/2021_GCP_GCCSA_for_NSW_short-header.zip
python -m src.population.census_tablebuilder --file data/raw/tablebuilder_incp_lfsp_age5p_nsw.xlsx --file data/raw/tablebuilder_rlhp_incp_nsw.xlsx
python -m src.population.build --tables data/processed/tables --n 2000
```

Layer B (households and spending):

```
python -m src.population.households --file data/raw/tablebuilder_hind_hcfmd_nprd_nsw.xlsx --rlhp data/raw/tablebuilder_rlhp_incp_nsw.xlsx
python -m src.population.hes --zip data/raw/hes_2015-16_all_data_cubes.zip
python -m src.population.cpi --file data/raw/cpi_table18_sep2026.xlsx
python -m src.population.layer_b
```

Layer C (behaviour and personality):

```
python -m src.population.layer_c
```

Outputs:

- `data/processed/agents_layer_a.csv` - one row per agent, demographics
- `data/processed/agents_layer_b.csv` - the same agents plus household and spending
- `data/processed/agents.csv` - **the complete population** (Layers A + B + C)
- `outputs/validation/marginals.png` - census vs synthetic, per attribute
- `outputs/validation/table_fit.csv`, `marginal_fit.csv` - Layer A fit
- `outputs/validation/layer_b_checks.csv` - Layer B checks
- `outputs/validation/layer_c_checks.csv`, `layer_c_traits.png` - Layer C checks

## Data sources

Both from the Australian Bureau of Statistics, 2021 Census. Save downloads in
`data/raw/` (not committed to git).

1. **DataPack**: General Community Profile, Greater Capital City Statistical
   Areas, New South Wales.
   https://www.abs.gov.au/census/find-census-data/datapacks
2. **TableBuilder exports** (free account needed), from the dataset "2021
   Census - employment, income and education", filter STATE (UR) = New South
   Wales, downloaded as Excel. https://tablebuilder.abs.gov.au
   - `tablebuilder_incp_lfsp_age5p_nsw.xlsx`: rows INCP, columns LFSP, wafers AGE5P
   - `tablebuilder_rlhp_incp_nsw.xlsx`: rows RLHP (all categories), columns INCP
   - `tablebuilder_hind_hcfmd_nprd_nsw.xlsx`: from the dataset "selected dwelling
     characteristics", filter STATE (EN) = New South Wales, rows HIND, columns
     HCFMD (2-digit level), wafers NPRD
3. **Household Expenditure Survey 2015-16**, "All data cubes" zip.
   https://www.abs.gov.au/statistics/economy/finance/household-expenditure-survey-australia-summary-results/latest-release
4. **Consumer Price Index**, Table 18 (quarterly index numbers by group,
   sub-group and expenditure class), saved as `cpi_table18_sep2026.xlsx`.
   https://www.abs.gov.au/statistics/economy/price-indexes-and-inflation/consumer-price-index-australia/latest-release

| Table | Census source | What it gives |
|---|---|---|
| `location_age_sex` | G04 Age by sex | Population base, people aged 15+ |
| `location_age_sex_income` | G17 Total personal income (weekly) by age by sex | Income |
| `location_age_sex_work` | G46 Labour force status by age by sex | Work status |
| `location_age_sex_living` | G27 Relationship in household by age by sex | Living arrangement |
| `age_income_work` | TableBuilder INCP x LFSP x AGE5P | How income and work status go together at each age |
| `living_income` | TableBuilder RLHP x INCP | How living arrangement and income go together |

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

## How Layer B works

1. **Household.** Each agent is placed in a census household that fits their
   living arrangement (a partnered agent in a couple household, and so on),
   with household income never below the agent's own income. Details and
   approximations are at the top of `src/population/households.py`.
2. **Income level.** Household income is adjusted for household size
   (equivalised) and ranked into five groups using cut-offs calculated from all
   NSW households.
3. **Spending.** The Household Expenditure Survey gives average weekly spending
   per household for each of those five groups. It is divided by the average
   number of adults per household, then scaled from 2015-16 prices to current
   prices with the CPI series for that category.

## How Layer C works

Every number lives in `config/layer_c.yaml`, marked as data or assumption.

| Attribute | Basis |
|---|---|
| Personal spending (varies around the Layer B group average, mean kept) | Assumption |
| `price_sensitivity` (0-1, higher for lower incomes) | Assumption |
| `novelty_seeking` (0-1, higher for younger people) | Assumption |
| `habit_loyalty` (0-1, higher for older people, negatively linked to novelty) | Assumption |
| `word_of_mouth_reach` (people told per month, heavy-tailed) | Assumption |
| `uses_delivery_apps` | Roy Morgan 2022 usage by generation and state, plus an assumed Sydney/regional gap |
| `diet_vegetarian` | Roy Morgan 2018 (12.1%), plus assumed age and location pattern |
| `diet_halal` | 2021 Census religion (G14) by location, plus an assumed share who look for halal food |

To test how much an assumption matters, copy the config, change it, and build
with `python -m src.population.layer_c --config <your copy>`. The formal
sensitivity analysis comes in Phase 6, once the simulation exists.

## Known limitations

Layer A:

- **The links between income, work and living arrangement are the same for
  both sexes and both locations.** The TableBuilder tables are not broken down
  by sex or location.
- **Living arrangement and work status are linked only through income, age,
  sex and location**, not directly.
- **Location has two categories** (Greater Sydney, rest of NSW), not the four in
  the project plan.
- **Living arrangement** describes the person's place in the household
  (partnered, lone parent, living with parents, and so on).

Layer B:

- **Spending is an average, not a personal figure.** Every agent in the same
  income group has the same spending. Variation between people is a Layer C
  assumption.
- **Spending depends on income level only.** Age, location and household type
  do not shift it yet, although the survey shows they matter.
- **The spending survey is from 2015-16 and Australia-wide.** Prices are
  brought up to date with the CPI, but changes in habits since then (delivery
  apps, working from home) are not captured.
- **Household assignment is approximate.** Households with two or more families
  are left out, relatives living with a couple or single parent are placed in
  "other family" households, and children's ages within a household are
  averaged. See `src/population/households.py`.
- **Household income is before tax**, while the survey ranks households by
  income after tax.

Layer C:

- **The personality traits are assumptions**, chosen to have sensible
  directions (lower income, more price sensitive), not measured values.
- **Diet and delivery-app use do not depend on income**, and halal depends on
  location only.
- **Delivery-app figures are from early 2022**, during the pandemic peak.

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

## Effect of the TableBuilder tables

Share of each group in a weekly income band, in the fitted population:

| | Community Profile tables only | With TableBuilder tables | Census (direct) |
|---|---|---|---|
| Full-time workers with nil or negative income | 6.9% | 0.4% | 0.4% |
| Full-time workers on $2000+ | 20.1% | 31.9% | 32.2% |
| Adults living with parents on $2000+ | 6.7% | 2.7% | 2.7% |
| Group household members on $2000+ | 13.9% | 7.6% | 7.6% |
