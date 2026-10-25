import sqlite3
import pandas as pd

con = sqlite3.connect("data/instamart.db")
orders = pd.read_sql_query("SELECT * FROM orders", con)
con.close()

orders["order_ts"] = pd.to_datetime(orders["order_ts"])
completed = orders[orders["status"] == "completed"]

snapshot = pd.Timestamp("2025-10-01 00:00:00")

rfm = completed.groupby("customer_id").agg(
    last_order_ts=("order_ts", "max"),
    frequency=("order_id", "count"),
    monetary=("order_amount", "sum"),
).reset_index()
rfm["recency"] = (snapshot - rfm["last_order_ts"]).dt.days

print("customers:", len(rfm))
print("recency median:", rfm["recency"].median())
print("recency min:", rfm["recency"].min())
print("frequency mean:", round(rfm["frequency"].mean(), 2), "max:", rfm["frequency"].max())
print("monetary mean:", round(rfm["monetary"].mean(), 2), "max:", round(rfm["monetary"].max(), 2))