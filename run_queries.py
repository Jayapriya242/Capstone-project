import sqlite3
import pandas as pd

con = sqlite3.connect("data/instamart.db")
text = open("sql/queries.sql", encoding="utf-8").read()

for block in text.split("-- Q")[1:]:
    title, sql = block.split("\n", 1)
    df = pd.read_sql_query(sql, con)
    print(f"\nQ{title.strip()}: {len(df)} rows")
    print(df.head(5).to_string(index=False))

con.close()