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
import random
import time
from mock_llm import chat, reset, TransientLLMError

BASE_DELAY = 0.05
MAX_RETRIES = 3

def call_with_retries(messages, ticket_id):
    for n in range(MAX_RETRIES + 1):
        try:
            return chat(messages)
        except TransientLLMError:
            if n == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** n * random.uniform(0.9, 1.15)
            print(f"retry ticket={ticket_id} attempt={n + 1} wait={wait:.4f}s")
            time.sleep(wait)

reset()
import json

ALLOWED = set(CATEGORIES)

def validate(reply):
    try:
        data = json.loads(reply)
        if isinstance(data, dict) and data.get("category") in ALLOWED:
            return "classified_json", data["category"]
        return "flagged_invalid_output", None
    except json.JSONDecodeError:
        pass
    words = set(re.findall(r"[a-z]+", reply.lower())) & ALLOWED
    if len(words) == 1:
        return "classified_keyword", words.pop()
    return "flagged_invalid_output", None

reset()
rows = []
for _, row in tickets[tickets["status"] == "ok"].iterrows():
    messages = [
        {"role": "system", "content": ZERO_SHOT},
        {"role": "user", "content": row["redacted"]},
    ]
    resp = call_with_retries(messages, row["ticket_id"])
    status, category = validate(resp.choices[0].message.content)
    rows.append({"ticket_id": row["ticket_id"], "status": status,
                 "category": category, "tokens": resp.usage.total_tokens})

results = pd.DataFrame(rows)
print(results["status"].value_counts())
print("total tokens:", results["tokens"].sum())
























