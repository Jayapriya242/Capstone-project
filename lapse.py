import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

con = sqlite3.connect("data/instamart.db")
orders = pd.read_sql_query("SELECT * FROM orders", con)
customers = pd.read_sql_query("SELECT * FROM customers", con)
products = pd.read_sql_query("SELECT * FROM products", con)
con.close()

orders["order_ts"] = pd.to_datetime(orders["order_ts"])
customers["signup_date"] = pd.to_datetime(customers["signup_date"])
cutoff = pd.Timestamp("2025-09-01")

# Only orders before the cutoff may be used for features
before = orders[orders["order_ts"] < cutoff]
done_before = before[before["status"] == "completed"]

# Cohort: at least one completed order before the cutoff
cohort_ids = done_before["customer_id"].unique()
feat = pd.DataFrame(index=pd.Index(cohort_ids, name="customer_id"))

# Label: 1 if no completed order from 2025-09-01 to 2025-09-30
window = orders[(orders["order_ts"] >= cutoff)
                & (orders["order_ts"] < pd.Timestamp("2025-10-01"))
                & (orders["status"] == "completed")]
ordered_in_window = set(window["customer_id"])
feat["lapsed"] = [0 if cid in ordered_in_window else 1 for cid in feat.index]

# Order-based features
last = done_before.groupby("customer_id")["order_ts"].max()
feat["recency"] = (cutoff - last).dt.days
feat["frequency"] = done_before.groupby("customer_id")["order_id"].count()
feat["monetary"] = done_before.groupby("customer_id")["order_amount"].sum()
feat["avg_discount"] = done_before.groupby("customer_id")["discount_pct"].mean()
feat["avg_delivery"] = done_before.groupby("customer_id")["delivery_minutes"].mean()

final = before[before["status"].isin(["completed", "cancelled"])]
feat["cancel_rate"] = final.groupby("customer_id")["status"].apply(lambda s: (s == "cancelled").mean())

with_cat = done_before.merge(products[["product_id", "category"]], on="product_id")
feat["top_category"] = with_cat.groupby("customer_id")["category"].agg(lambda s: s.mode().iloc[0])

# Customer attributes
cust = customers.set_index("customer_id")
feat["city"] = cust["city"]
feat["age"] = cust["age"]
feat["gender"] = cust["gender"]
feat["tenure"] = (cutoff - cust["signup_date"]).dt.days

print("cohort:", len(feat))
print("lapsed:", feat["lapsed"].sum(), round(feat["lapsed"].mean() * 100, 2), "%")
print("features:", list(feat.columns.drop("lapsed")))
print("missing values:", feat.isna().sum().sum())

X = feat.drop(columns=["lapsed"])
y = feat["lapsed"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
print("train rows:", len(X_train), "test rows:", len(X_test))

baseline = [0] * len(y_test)
print("baseline accuracy:", round((y_test == 0).mean() * 100, 1), "%")
print("baseline lapsed F1:", f1_score(y_test, baseline, zero_division=0))



from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

cat_cols = ["city", "gender", "top_category"]
num_cols = ["recency", "frequency", "monetary", "avg_discount",
            "avg_delivery", "cancel_rate", "age", "tenure"]

prep = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
    ("num", StandardScaler(), num_cols),
])

X_train_t = prep.fit_transform(X_train)   # learns from training rows only
X_test_t = prep.transform(X_test)         # applies the same settings

print("train shape:", X_train_t.shape, "test shape:", X_test_t.shape)
print("missing in train:", pd.isna(X_train_t).sum(), "missing in test:", pd.isna(X_test_t).sum())

scaler = prep.named_transformers_["num"]
freq_idx = num_cols.index("frequency")
print("frequency mean learned by scaler (train):", round(scaler.mean_[freq_idx], 3))
print("frequency mean across full cohort:", round(X["frequency"].mean(), 3))