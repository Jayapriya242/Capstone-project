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

STATE_DIR = "state"

def _checkpoint(run_id, scenario_id, messages, steps, llm_calls, tokens_used):
    ck = {"run_id": run_id, "scenario_id": scenario_id, "messages": messages,
          "tokens_used": tokens_used, "llm_calls": llm_calls,
          "step": len(steps), "tools_executed": [s["tool"] for s in steps],
          "steps": steps}
    with open(os.path.join(STATE_DIR, f"{run_id}.json"), "w", encoding="utf-8") as f:
        json.dump(ck, f, ensure_ascii=False)

def _loop(run_id, scenario_id, messages, steps, llm_calls, tokens_used,
          max_iterations, token_limit, time_limit_seconds, crash_after):
    import time
    from mock_llm import chat
    start = time.monotonic()
    while llm_calls < max_iterations:
        if time.monotonic() - start >= time_limit_seconds:
            return {"status": "FALLBACK_TIMEOUT",
                    "final_answer": "Stopped at the time limit. Please hand off to a human agent.",
                    "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used}
        resp = chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += resp.usage.total_tokens
        if tokens_used > token_limit:
            return {"status": "FALLBACK_TOKEN_LIMIT",
                    "final_answer": "Stopped at the token limit. Please hand off to a human agent.",
                    "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used}
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return {"status": msg.content.split(":")[0], "final_answer": msg.content,
                    "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used}
        messages.append({"role": "assistant", "content": None,
                         "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                                        for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            result = TOOL_MAP[tc.name](**tc.arguments)
            steps.append({"tool": tc.name, "arguments": tc.arguments, "observation": result})
            messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                             "content": json.dumps(result)})
            _checkpoint(run_id, scenario_id, messages, steps, llm_calls, tokens_used)
            if crash_after is not None and len(steps) == crash_after:
                raise RuntimeError(f"simulated crash after tool execution {len(steps)}")
    return {"status": "FALLBACK_MAX_ITERATIONS",
            "final_answer": "Reached the step limit. Please hand off to a human agent.",
            "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used}

def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000,
              time_limit_seconds=10, run_id="run", crash_after=None):
    system = ("AGENT. You are a refund agent. Look up the order, check the refund policy, "
              "then calculate, issue or escalate the refund. Use the tools provided.")
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user_message}]
    return _loop(run_id, scenario_id, messages, [], 0, 0,
                 max_iterations, token_limit, time_limit_seconds, crash_after)

def resume_agent(run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    with open(os.path.join(STATE_DIR, f"{run_id}.json"), encoding="utf-8") as f:
        ck = json.load(f)
    print("resuming", run_id, "after tools:", ck["tools_executed"])
    return _loop(ck["run_id"], ck["scenario_id"], ck["messages"], ck["steps"],
                 ck["llm_calls"], ck["tokens_used"],
                 max_iterations, token_limit, time_limit_seconds, None)

if __name__ == "__main__":
    print("tools defined:", list(TOOL_MAP))
    print(get_order("ORD-999999"))
    print(get_order(pd.read_sql_query("SELECT order_id FROM orders LIMIT 1",
                    sqlite3.connect("data/instamart.db"))["order_id"][0]))
    result = run_agent("Please refund order ORD-110093, it arrived late.", "TEST")
    print(result["status"], result["llm_calls"], result["tokens_used"])
    print(result["final_answer"])
    print(run_agent("Please refund order ORD-110093, it arrived late.", "TEST", token_limit=200)["status"])
    print(run_agent("Please refund order ORD-110093, it arrived late.", "TEST", time_limit_seconds=0)["status"])
    loop = run_agent("[LOOP] Refund my order.", "TEST")
    print(loop["status"], loop["llm_calls"])

    scen = pd.read_csv("data/raw/agent_scenarios.csv")
    s1_msg = scen.loc[scen["scenario_id"] == "S1", "message"].iloc[0]
    try:
        run_agent(s1_msg, "S1", run_id="S1_demo", crash_after=2)
    except RuntimeError as e:
        print("crash:", e)
    res = resume_agent("S1_demo")
    print("final:", res["status"], "llm_calls:", res["llm_calls"], "tokens:", res["tokens_used"])
    print("tools across crash and resume:", [s["tool"] for s in res["steps"]])
    print("final answer:", res["final_answer"])








