import pandas as pd
from mock_llm import est_tokens

MAX_INPUT_TOKENS = 400

tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
tickets["status"] = "ok"

empty = tickets["text"].str.strip() == ""
tickets.loc[empty, "status"] = "skipped_empty"

tokens = tickets["text"].apply(est_tokens)
long = (~empty) & (tokens > MAX_INPUT_TOKENS)
tickets.loc[long, "status"] = "flagged_long"

print(tickets["status"].value_counts())
print("estimated tokens in flagged_long tickets:", tokens[long].sum())