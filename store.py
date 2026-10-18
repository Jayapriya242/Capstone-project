import sqlite3
import pandas as pd

customers = pd.read_csv("data/clean/customers.csv")
orders = pd.read_csv("data/clean/orders.csv")
products = pd.read_csv("data/raw/products.csv")
products.to_csv("data/clean/products.csv", index=False)

con = sqlite3.connect("data/instamart.db")
customers.to_sql("customers", con, if_exists="replace", index=False)
products.to_sql("products", con, if_exists="replace", index=False)
orders.to_sql("orders", con, if_exists="replace", index=False)
con.close()

print("stored: customers", len(customers), "products", len(products), "orders", len(orders))