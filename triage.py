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
def run_pipeline(prompt, label):
    reset()
    rows = []
    for _, row in tickets[tickets["status"] == "ok"].iterrows():
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": row["redacted"]},
        ]
        resp = call_with_retries(messages, row["ticket_id"])
        status, category = validate(resp.choices[0].message.content)
        rows.append({"ticket_id": row["ticket_id"], "status": status,
                     "category": category, "tokens": resp.usage.total_tokens,
                     "true_category": row["true_category"]})
    df = pd.DataFrame(rows)
    classified = df[df["status"].str.startswith("classified")]
    correct = (classified["category"] == classified["true_category"]).sum()
    total_tokens = df["tokens"].sum()
    print()
    print(label)
    print("classified:", len(classified))
    print("correct:", correct, "of", len(classified))
    print("accuracy:", round(correct / len(classified) * 100, 1), "%")
    print("total tokens:", total_tokens)
    print("tokens per classified ticket:", round(total_tokens / len(classified), 1))
    return df, classified

zs_df, zs_cls = run_pipeline(ZERO_SHOT, "zero-shot")
fs_df, fs_cls = run_pipeline(FEW_SHOT, "few-shot")

print()
print("zero-shot confusion table (rows = true, columns = predicted):")
print(pd.crosstab(zs_cls["true_category"], zs_cls["category"]))

import os
os.makedirs("outputs", exist_ok=True)

TOKEN_BUDGET_DEMO = 6000
TOKEN_BUDGET_FULL = 100000

def run_budgeted(prompt, budget):
    reset()
    spent = 0
    rows = []
    for _, row in tickets.iterrows():
        base = {"ticket_id": row["ticket_id"], "customer_id": row["customer_id"],
                "true_category": row["true_category"]}
        if row["status"] != "ok":
            rows.append({**base, "status": row["status"], "category": None, "tokens_used": 0})
            continue
        messages = [{"role": "system", "content": prompt},
                    {"role": "user", "content": row["redacted"]}]
        estimate = sum(est_tokens(m["content"]) for m in messages)
        if spent + estimate > budget:
            rows.append({**base, "status": "budget_exhausted", "category": None, "tokens_used": 0})
            continue
        resp = call_with_retries(messages, row["ticket_id"])
        status, category = validate(resp.choices[0].message.content)
        used = resp.usage.total_tokens
        spent += used
        rows.append({**base, "status": status, "category": category, "tokens_used": used})
    return pd.DataFrame(rows), spent

# (a) Budget demo with zero-shot prompt
demo_df, demo_spent = run_budgeted(ZERO_SHOT, TOKEN_BUDGET_DEMO)
demo_df.to_csv("outputs/triage_budget_demo.csv", index=False)
print("demo tokens spent:", demo_spent)
print("demo status counts:")
print(demo_df["status"].value_counts())

# (b) Full run with few-shot prompt, then routing
full_df, full_spent = run_budgeted(FEW_SHOT, TOKEN_BUDGET_FULL)

seg = pd.read_csv("outputs/customer_segments.csv").set_index("customer_id")
full_df["segment_name"] = full_df["customer_id"].map(seg["segment_name"])
full_df["premium"] = full_df["customer_id"].map(seg["is_premium"]).fillna(False).astype(bool)

def route(row):
    if row["status"] in ("skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"):
        return "human_review", "NORMAL"
    if row["category"] == "fraud":
        return "security_team", "HIGH"
    if row["premium"]:
        return "premium_desk", "HIGH"
    return "support_queue", "NORMAL"

routes = full_df.apply(route, axis=1, result_type="expand")
full_df["route"] = routes[0]
full_df["priority"] = routes[1]

out = full_df[["ticket_id", "customer_id", "status", "category", "segment_name",
               "priority", "route", "tokens_used"]]
out.to_csv("outputs/triage_results.csv", index=False)

print("rows:", len(out))
print("full run tokens spent:", full_spent)
print(out["status"].value_counts())
print(out["route"].value_counts())
print("budget_exhausted in full run:", (out["status"] == "budget_exhausted").sum())





















