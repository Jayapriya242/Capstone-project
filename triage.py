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

import re

EMAIL = r"[\w.+-]+@[\w-]+\.[\w.]+"
PHONE = r"\b[6-9]\d{9}\b"
ORDER = r"ORD-\d{6}"
PATTERNS = [(EMAIL, "[EMAIL]"), (PHONE, "[PHONE]"), (ORDER, "[ORDER_ID]")]

def redact(text):
    counts = {}
    for pattern, placeholder in PATTERNS:
        text, n = re.subn(pattern, placeholder, text)
        counts[placeholder] = n
    return text, counts

results = tickets["text"].apply(redact)
tickets["redacted"] = results.apply(lambda r: r[0])
totals = {p: sum(r[1][p] for r in results) for _, p in PATTERNS}
print("substitutions:", totals, "total:", sum(totals.values()))

# Check: re-applying the patterns to the redacted text should find nothing
left = sum(len(re.findall(p, t)) for t in tickets["redacted"] for p, _ in PATTERNS)
print("left after redaction:", left)


























