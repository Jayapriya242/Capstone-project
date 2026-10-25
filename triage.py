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
CATEGORIES = ["billing", "delivery", "technical", "fraud"]

ZERO_SHOT = (
    "CLASSIFY the customer support ticket into exactly one category: "
    + ", ".join(CATEGORIES)
    + ". Reply with JSON only, in the form "
    '{"category": "...", "reason": "..."}.'
)

FEW_SHOT = ZERO_SHOT + (
    "\n\nEXAMPLES:\n"
    'Ticket: "I was charged twice for my order." -> '
    '{"category": "billing", "reason": "duplicate charge"}\n'
    'Ticket: "An unauthorized payment appeared on my card." -> '
    '{"category": "fraud", "reason": "unauthorized transaction"}\n'
)

print("zero-shot contains CLASSIFY:", "CLASSIFY" in ZERO_SHOT)
print("zero-shot contains EXAMPLES:", "EXAMPLES:" in ZERO_SHOT)
print("few-shot contains EXAMPLES:", "EXAMPLES:" in FEW_SHOT)

























