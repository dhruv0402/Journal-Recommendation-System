import pandas as pd
import zipfile
import os
import shutil
from pathlib import Path

# Resolve project root dynamically so the script works from any working directory
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

rows = []


def clean_journal(name):
    return name.replace("_", " ").strip()


def process_csv(path, journal):
    df = pd.read_csv(path)

    if "title" in df.columns:
        df = df.rename(columns={"title": "article_title"})

    df["journal_name"] = journal

    return df[
        ["journal_name", "article_title", "abstract", "article_url", "reference_titles"]
    ]


# 1️⃣ Load existing dataset
master = pd.read_csv(DATA_DIR / "master_journals.csv")

master = master[
    ["journal_name", "article_title", "abstract", "article_url", "reference_titles"]
]

rows.append(master)

# 2️⃣ Extract journal_project.zip
with zipfile.ZipFile(DATA_DIR / "journal_project.zip") as z:
    z.extractall("journals_temp")

for root, dirs, files in os.walk("journals_temp"):
    for f in files:
        if f.endswith(".csv"):
            path = os.path.join(root, f)
            journal = clean_journal(os.path.basename(os.path.dirname(path)))

            try:
                df = process_csv(path, journal)
                rows.append(df)
            except Exception as e:
                print("Error processing:", path, e)

# 3️⃣ Extract JNCA zip
with zipfile.ZipFile(
    DATA_DIR / "Journal_of_Network_and_Computer_Applications.zip"
) as z:
    z.extractall("jnca_temp")

for f in os.listdir("jnca_temp"):
    if f.endswith(".csv"):
        path = os.path.join("jnca_temp", f)

        df = process_csv(path, "Journal of Network and Computer Applications")
        rows.append(df)

# 4️⃣ Merge everything
merged = pd.concat(rows, ignore_index=True)

# 5️⃣ Cleaning
merged = merged.dropna(subset=["article_title", "abstract"])
merged["abstract"] = merged["abstract"].astype(str)
merged = merged[merged["abstract"].str.len() > 30]

merged["article_title"] = merged["article_title"].str.strip()
merged["abstract"] = merged["abstract"].str.strip()

merged = merged.drop_duplicates(subset=["article_title", "abstract"])

# 6️⃣ Save
merged.to_csv(DATA_DIR / "master_journals_expanded.csv", index=False)

print("Final dataset size:", len(merged))

# Cleanup temporary folders
if os.path.exists("journals_temp"):
    shutil.rmtree("journals_temp")

if os.path.exists("jnca_temp"):
    shutil.rmtree("jnca_temp")
