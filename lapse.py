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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score

lr_grid = GridSearchCV(
    LogisticRegression(max_iter=1000),
    {"C": [0.01, 0.1, 1, 10, 100]},
    cv=5, scoring="f1",
).fit(X_train_t, y_train)
print("LR best params:", lr_grid.best_params_, "best CV F1:", round(lr_grid.best_score_, 3))
rf_grid = GridSearchCV(
    RandomForestClassifier(n_estimators=100, random_state=42),
    {"max_depth": [3, 5, 8], "min_samples_leaf": [5, 10, 20]},
    cv=5, scoring="f1",
).fit(X_train_t, y_train)
print("RF best params:", rf_grid.best_params_, "best CV F1:", round(rf_grid.best_score_, 3))
for name, model in [("LR", lr_grid.best_estimator_), ("RF", rf_grid.best_estimator_)]:
    pred = model.predict(X_test_t)
    print(name, "confusion matrix:\n", confusion_matrix(y_test, pred))
    print(name, "precision:", round(precision_score(y_test, pred), 3),
          "recall:", round(recall_score(y_test, pred), 3),
          "F1:", round(f1_score(y_test, pred), 3))

import numpy as np
from sklearn.utils import resample

y_arr = y_train.values
lap_idx = np.where(y_arr == 1)[0]
non_idx = np.where(y_arr == 0)[0]
print("before balancing: lapsed", len(lap_idx), "non-lapsed", len(non_idx))

# Keep every original training row, and add random duplicates of lapsed rows
extra = resample(lap_idx, replace=True,
                 n_samples=len(non_idx) - len(lap_idx), random_state=42)
bal_idx = np.concatenate([non_idx, lap_idx, extra])
X_bal, y_bal = X_train_t[bal_idx], y_arr[bal_idx]
print("after balancing: lapsed", (y_bal == 1).sum(), "non-lapsed", (y_bal == 0).sum())

bal_model = LogisticRegression(C=0.1, max_iter=1000).fit(X_bal, y_bal)
pred = bal_model.predict(X_test_t)
print("balanced LR confusion matrix:\n", confusion_matrix(y_test, pred))
print("balanced LR precision:", round(precision_score(y_test, pred), 3),
      "recall:", round(recall_score(y_test, pred), 3),
      "F1:", round(f1_score(y_test, pred), 3))
















