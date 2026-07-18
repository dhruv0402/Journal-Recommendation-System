import pandas as pd

df = pd.read_csv("data/master_journals_expanded.csv")

print("Rows:", len(df))
print("Journals:", df["journal_name"].nunique())
print("Missing abstracts:", df["abstract"].isna().sum())
print("Duplicates:", df.duplicated(subset=["article_title", "abstract"]).sum())
