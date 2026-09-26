JAL-SETU DUMMY DATASET

This dataset matches the current jalsetu.db schema that was inspected.

There are 7 CSV files because the database is relational/normalized:
- pumps.csv
- mechanics.csv
- inspections.csv
- diagnoses.csv
- repairs.csv
- verifications.csv
- audit_log.csv

Do NOT put all of these into one table. Foreign-key relationships connect them.

For a safe import test:
1. Put this folder and import_dummy_dataset.py in the same folder as jalsetu(1).db.
2. Run: python import_dummy_dataset.py
3. The script creates jalsetu_import_test.db and leaves the original database unchanged.

Relationship chain:
pumps -> inspections -> diagnoses
pumps -> repairs -> mechanics
inspections -> repairs
repairs -> verifications
pumps/inspections -> audit_log
