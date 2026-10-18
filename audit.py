import pandas as pd

orders = pd.read_csv("data/raw/orders.csv")
customers = pd.read_csv("data/raw/customers.csv")
tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)

print(orders.shape, customers.shape, tickets.shape)

print(orders.dtypes)
print(orders.isna().sum())
print("raw rows:", len(orders))
print("exact duplicates:", orders.duplicated().sum())
print("pending rows:", (orders["status"] == "pending").sum())
print("distinct order_id:", orders["order_id"].nunique())
amount_num = pd.to_numeric(orders["order_amount"], errors="coerce")
print("non-numeric amounts:", amount_num.isna().sum())
print("negative delivery:", (orders["delivery_minutes"] < 0).sum())
print("blank age:", customers["age"].isna().sum())
print("age outside 10-100:", ((customers["age"] < 10) | (customers["age"] > 100)).sum())
print("raw city spellings:", customers["city"].nunique())
print("raw gender spellings:", customers["gender"].nunique())
print("blank gender:", customers["gender"].isna().sum())