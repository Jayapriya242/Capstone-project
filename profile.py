import sqlite3
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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

X = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
km = KMeans(n_clusters=4, n_init=10, random_state=42).fit(X)
rfm["segment_id"] = km.labels_

print(rfm["segment_id"].value_counts().sort_index())

profile = rfm.groupby("segment_id")[["recency", "frequency", "monetary"]].mean().round(2)
profile["size"] = rfm["segment_id"].value_counts().sort_index()
print(profile)
premium_id = profile["monetary"].idxmax()
rfm["is_premium"] = rfm["segment_id"] == premium_id
from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit(X)
print(pca.explained_variance_ratio_)
segment_names = {
    0: "Occasional low-spend buyers",
    1: "High-value regulars (Premium)",
    2: "Loyal frequent buyers",
    3: "Dormant / lapsed customers",
}
rfm["segment_name"] = rfm["segment_id"].map(segment_names)
rfm[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]].to_csv("outputs/customer_segments.csv", index=False)