import sqlite3
import pandas as pd

con = sqlite3.connect("data/instamart.db")
customers = pd.read_sql_query("SELECT * FROM customers", con)
orders = pd.read_sql_query("SELECT * FROM orders", con)
con.close()

completed = orders[orders["status"] == "completed"]

checks = [
    (len(customers) == 1200, "customer rows"),
    (len(orders) == 12309, "order rows"),
    (orders["order_id"].nunique() == 12309, "one row per order"),
    (orders["status"].value_counts().to_dict() == {"completed": 11110, "cancelled": 1199}, "status counts"),
    (customers["city"].value_counts()["Pune"] == 227, "Pune count"),
    (customers["gender"].value_counts()["unknown"] == 30, "unknown gender count"),
    (customers["age"].isna().sum() == 0, "no missing ages"),
    (orders["order_amount"].isna().sum() == 0, "no missing amounts"),
    (round(completed["order_amount"].sum(), 2) == 2998668.80, "completed revenue"),
]

failed = [name for ok, name in checks if not ok]
for ok, name in checks:
    print("PASS" if ok else "FAIL", name)

if failed:
    raise SystemExit("FAILED: " + ", ".join(failed))
print("ALL CHECKPOINTS PASSED")