import sqlite3
from pathlib import Path
import shutil

SOURCE_DB = "jalsetu.db"
TEST_DB = "jalsetu_import_test.db"
DATA_DIR = Path("jalsetu_dummy_dataset/jalsetu_dummy_dataset")

shutil.copy2(SOURCE_DB, TEST_DB)

conn = sqlite3.connect(TEST_DB)
conn.execute("PRAGMA foreign_keys = ON")
cur = conn.cursor()

# Insert parent tables first, then dependent tables.
tables_in_order = [
    "pumps",
    "mechanics",
    "inspections",
    "diagnoses",
    "repairs",
    "verifications",
    "audit_log",
]

for table in tables_in_order:
    import csv
    with open(DATA_DIR / f"{table}.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        placeholders = ",".join(["?"] * len(columns))
        sql = f'INSERT INTO "{table}" ({",".join(columns)}) VALUES ({placeholders})'
        for row in reader:
            cur.execute(sql, [row[c] if row[c] != "" else None for c in columns])

conn.commit()

# Quick verification
for table in tables_in_order:
    count = cur.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
    print(f"{table}: {count} rows")

conn.close()
print(f"\nDone. Your original {SOURCE_DB} was not modified.")
print(f"Test database created: {TEST_DB}")
