import json
import sqlite3
import os
from datetime import datetime
import pandas as pd
from rag import retrieve, chunks

os.makedirs("outputs", exist_ok=True)
os.makedirs("state", exist_ok=True)
LEDGER = "outputs/refund_ledger.jsonl"

def get_order(order_id):
    con = sqlite3.connect("data/instamart.db")
    df = pd.read_sql_query("SELECT order_id, status, order_amount, delivery_minutes "
                           "FROM orders WHERE order_id = ?", con, params=(order_id,))
    con.close()
    if df.empty:
        return {"error": "order not found"}
    row = df.iloc[0]
    return {"order_id": row["order_id"], "status": row["status"],
            "order_amount": float(row["order_amount"]),
            "delivery_minutes": None if pd.isna(row["delivery_minutes"]) else float(row["delivery_minutes"])}

def search_policy(query):
    hits = retrieve(query, k=3)
    return {"results": "\n".join(f"[{h['source']}] {h['text']}" for h in hits)}

def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is not None and delivery_minutes > 45:
        return {"refund_inr": round(float(order_amount), 2)}
    return {"refund_inr": 0.0}

def _log(tool, payload):
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"),
                            "tool": tool, **payload}) + "\n")

def issue_refund(order_id, refund_inr):
    _log("issue_refund", {"order_id": order_id, "refund_inr": refund_inr})
    return {"refund_inr": refund_inr, "result": "issued"}

def escalate_to_human(order_id, refund_inr):
    _log("escalate_to_human", {"order_id": order_id, "refund_inr": refund_inr})
    return {"refund_inr": refund_inr, "result": "escalated"}

TOOLS_SPEC = [
    {"name": "get_order", "description": "Look up an order's status, amount and delivery time.",
     "parameters": {"order_id": "string"}},
    {"name": "search_policy", "description": "Search the policy documents for text relevant to a query.",
     "parameters": {"query": "string"}},
    {"name": "calculate_refund", "description": "Compute the refund amount from the order amount and delivery time.",
     "parameters": {"order_amount": "number", "delivery_minutes": "number or null"}},
    {"name": "issue_refund", "description": "Issue a refund of up to INR 1000 and record it in the ledger.",
     "parameters": {"order_id": "string", "refund_inr": "number"}},
    {"name": "escalate_to_human", "description": "Send a refund above INR 1000 to a human lead and record it.",
     "parameters": {"order_id": "string", "refund_inr": "number"}},
]

TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}

def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    import time
    from mock_llm import chat
    system = ("AGENT. You are a refund agent. Look up the order, check the refund policy, "
              "then calculate, issue or escalate the refund. Use the tools provided.")
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user_message}]
    steps, llm_calls, tokens_used = [], 0, 0
    start = time.monotonic()

    for _ in range(max_iterations):
        if time.monotonic() - start >= time_limit_seconds:
            return {"status": "FALLBACK_TIMEOUT", "final_answer": "Stopped at the time limit. Please hand off to a human agent.",
                    "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used}
        resp = chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += resp.usage.total_tokens
        if tokens_used > token_limit:
            return {"status": "FALLBACK_TOKEN_LIMIT", "final_answer": "Stopped at the token limit. Please hand off to a human agent.",
                    "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used}

        msg = resp.choices[0].message
        if not msg.tool_calls:
            text = msg.content
            return {"status": text.split(":")[0], "final_answer": text,
                    "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used}

        messages.append({"role": "assistant", "content": None,
                         "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                                        for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            result = TOOL_MAP[tc.name](**tc.arguments)
            steps.append({"tool": tc.name, "arguments": tc.arguments, "observation": result})
            messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                             "content": json.dumps(result)})

    return {"status": "FALLBACK_MAX_ITERATIONS", "final_answer": "Reached the step limit. Please hand off to a human agent.",
            "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used}


if __name__ == "__main__":
    print("tools defined:", list(TOOL_MAP))
    print(get_order("ORD-999999"))
    print(get_order(pd.read_sql_query("SELECT order_id FROM orders LIMIT 1",
                    sqlite3.connect("data/instamart.db"))["order_id"][0]))
    result = run_agent("Please refund order ORD-110093, it arrived late.", "TEST")
    print(result["status"], result["llm_calls"], result["tokens_used"])
    print(result["final_answer"])