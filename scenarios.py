import json
import pandas as pd
from agent import run_agent

scen = pd.read_csv("data/raw/agent_scenarios.csv", keep_default_na=False)

with open("outputs/agent_runs.jsonl", "w", encoding="utf-8") as out:
    for _, row in scen.iterrows():
        sid = row["scenario_id"]
        res = run_agent(row["message"], sid, run_id=sid)
        record = {
            "scenario_id": sid,
            "status": res["status"],
            "final_answer": res["final_answer"],
            "llm_calls": res["llm_calls"],
            "tokens_used": res["tokens_used"],
            "steps": res["steps"],
        }
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(sid, res["status"], res["llm_calls"], res["tokens_used"],
              res["final_answer"][:60])