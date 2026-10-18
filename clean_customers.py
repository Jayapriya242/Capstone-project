import pandas as pd

customers = pd.read_csv("data/raw/customers.csv")

# 1. City: strip spaces, then title case
customers["city"] = customers["city"].str.strip().str.title()
print("city counts:")
print(customers["city"].value_counts())

# 2. Gender: map spellings to male / female / unknown
gender_map = {"male": "male", "m": "male", "female": "female", "f": "female"}
customers["gender"] = customers["gender"].fillna("").str.strip().str.lower().map(gender_map).fillna("unknown")
print("gender counts:")
print(customers["gender"].value_counts())

# 3. Age: outside 10-100 becomes missing, then fill with the city median
customers.loc[(customers["age"] < 10) | (customers["age"] > 100), "age"] = None
print("missing ages before fill:", customers["age"].isna().sum())
customers["age"] = customers["age"].fillna(
    customers.groupby("city")["age"].transform("median"))
print("missing ages after fill:", customers["age"].isna().sum())
print("age min/max:", customers["age"].min(), customers["age"].max())
print("rows:", len(customers))