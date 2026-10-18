import sqlite3
import pandas as pd

con = sqlite3.connect("data/instamart.db")
blocks = open("sql/queries.sql", encoding="utf-8").read().split("-- Q")[1:]

def run(n):
    sql = next(b for b in blocks if b.startswith(f"{n} ")).split("\n", 1)[1]
    return pd.read_sql_query(sql, con)

q6 = run(6)
print("Q6 rank differs from row_num:", (q6["rnk"] != q6["row_num"]).sum())
print(q6[q6["row_num"] == 1][["city", "customer_id", "total_spend"]].to_string(index=False))

q7 = run(7)
print("Q7 last cumulative:", q7["cumulative_revenue"].iloc[-1])