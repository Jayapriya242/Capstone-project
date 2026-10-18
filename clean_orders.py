import pandas as pd

orders = pd.read_csv("data/raw/orders.csv")
orders["order_ts"] = pd.to_datetime(orders["order_ts"])
print("after timestamps:", len(orders))

orders = orders.drop_duplicates()
print("after exact duplicates:", len(orders))
orders = orders.sort_values("order_ts").drop_duplicates(subset="order_id", keep="last")
print("after latest per order:", len(orders))

products = pd.read_csv("data/raw/products.csv")
orders = orders.merge(products[["product_id", "price_inr"]], on="product_id", how="left")

orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
print("missing amounts before fill:", orders["order_amount"].isna().sum())

computed = (orders["price_inr"] * orders["quantity"] * (1 - orders["discount_pct"] / 100)).round(2)
orders["order_amount"] = orders["order_amount"].fillna(computed)
print("missing amounts after fill:", orders["order_amount"].isna().sum())
print("rows:", len(orders))

orders = orders.merge(products[["product_id", "category"]], on="product_id", how="left")

# Measurement 1: negative delivery times, counted before they are set to missing
print("negative delivery:", (orders["delivery_minutes"] < 0).sum())
orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = None

completed = orders["status"] == "completed"

# Measurement 2: completed orders missing a delivery time, counted before the median fill
print("completed missing before fill:",
      orders.loc[completed, "delivery_minutes"].isna().sum())

category_median = orders[completed].groupby("category")["delivery_minutes"].transform("median")
orders.loc[completed, "delivery_minutes"] = orders.loc[completed, "delivery_minutes"].fillna(category_median)

print("rows:", len(orders))
print(orders["status"].value_counts())
print("missing amounts:", orders["order_amount"].isna().sum())
print("missing delivery, completed:", orders.loc[completed, "delivery_minutes"].isna().sum())
print("missing delivery, cancelled:", orders.loc[~completed, "delivery_minutes"].isna().sum())
print(f"completed revenue: INR {orders.loc[completed, 'order_amount'].sum():,.2f}")

orders = orders.drop(columns=["price_inr", "category"])
orders["order_ts"] = orders["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
orders.to_csv("data/clean/orders.csv", index=False)