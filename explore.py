import numpy as np
import sqlite3
import pandas as pd
con = sqlite3.connect("data/instamart.db")
orders = pd.read_sql_query("SELECT * FROM orders", con)
customers = pd.read_sql_query("SELECT * FROM customers", con)
products = pd.read_sql_query("SELECT * FROM products", con)
con.close()

completed = orders[orders["status"] == "completed"]
print("completed orders:", len(completed))

joined = (completed
          .merge(customers[["customer_id", "city"]], on="customer_id")
          .merge(products[["product_id", "category"]], on="product_id"))

pivot = pd.pivot_table(joined, values="order_amount", index="city",
                       columns="category", aggfunc="sum", margins=True)
print(pivot.round(2))
print("grand total:", round(joined["order_amount"].sum(), 2))

city_report = joined.groupby("city").agg(
    total=("order_amount", "sum"),
    mean=("order_amount", "mean"),
    count=("order_amount", "count"),
).round(2)
print(city_report)
amt = completed["order_amount"]
print("mean:", round(amt.mean(), 2))
print("median:", amt.median())
print("skew:", round(amt.skew(), 2))

q1, q3 = amt.quantile(0.25), amt.quantile(0.75)
iqr = q3 - q1
upper, lower = q3 + 1.5 * iqr, q1 - 1.5 * iqr
print("Q1:", q1, "Q3:", q3, "IQR:", iqr)
print("upper fence:", upper, "lower fence:", lower)
print("outliers above:", (amt > upper).sum())
print("outliers below:", (amt < lower).sum())
cols = ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
corr = completed[cols].corr()
print(corr.round(3))

abs_corr = corr.abs()
mask = np.eye(len(cols), dtype=bool)
pairs = corr.abs().where(~mask).stack()
pair = pairs.idxmax()
print("largest absolute correlation:", pair, round(corr.loc[pair[0], pair[1]], 3))