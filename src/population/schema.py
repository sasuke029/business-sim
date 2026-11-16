"""Attribute definitions for Layer A (demographics).

Every agent attribute is categorical. The category lists here are the single
source of truth: input tables are checked against them, and the order here is
the order used in arrays and charts.

Categories follow what the 2021 Census General Community Profile publishes.
"""

CORE_ATTRIBUTES = {
    "location": ["greater_sydney", "rest_of_nsw"],
    "age": ["15-19", "20-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75-84", "85+"],
    "sex": ["female", "male"],
    # Weekly personal income
    "income": ["nil_or_negative", "$1-499", "$500-999", "$1000-1499",
               "$1500-1999", "$2000+"],
    "work": ["full_time", "part_time", "unemployed", "not_in_labour_force"],
    # The person's place in their household (census: relationship in household)
    "living": ["partnered", "lone_parent", "child_at_home", "other_in_family_home",
               "group_household", "lone_person"],
}

# Some census tables use broader bands than ours. A grouped column lets a table
# say "15-24" while the agents keep the finer "15-19" and "20-24".
# column name -> (attribute it describes, {broad band: [our categories]})
GROUPED_COLUMNS = {
    "age_broad": ("age", {
        "15-24": ["15-19", "20-24"],
        "25-34": ["25-34"], "35-44": ["35-44"], "45-54": ["45-54"],
        "55-64": ["55-64"], "65-74": ["65-74"], "75-84": ["75-84"], "85+": ["85+"],
    }),
}


def categories(attribute: str) -> list:
    if attribute not in CORE_ATTRIBUTES:
        raise KeyError(f"Unknown attribute '{attribute}'. Known: {list(CORE_ATTRIBUTES)}")
    return CORE_ATTRIBUTES[attribute]
