import json
import sqlite3
import os
import sys

DB_FILE = "jalsetu_import_test.db"
JSON_FILE = "jalsetu_test_data.json"

# Parent tables must be imported before child tables
IMPORT_ORDER = [
    "pumps",
    "mechanics",
    "inspections",
    "diagnoses",
    "repairs",
    "verifications",
    "audit_log"
]


def load_json():
    if not os.path.exists(JSON_FILE):
        print(f"ERROR: {JSON_FILE} was not found.")
        sys.exit(1)

    with open(JSON_FILE, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON: {e}")
            sys.exit(1)

    return data


def table_exists(cursor, table_name):
    result = cursor.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name = ?
        """,
        (table_name,)
    ).fetchone()

    return result is not None


def get_columns(cursor, table_name):
    rows = cursor.execute(
        f'PRAGMA table_info("{table_name}")'
    ).fetchall()

    return {row[1] for row in rows}


def import_table(cursor, table_name, rows):
    if not rows:
        print(f"{table_name}: no records to import")
        return 0

    if not table_exists(cursor, table_name):
        print(f"WARNING: Table '{table_name}' does not exist. Skipping.")
        return 0

    existing_columns = get_columns(cursor, table_name)

    inserted = 0
    skipped = 0

    for row in rows:
        if not isinstance(row, dict):
            print(f"WARNING: Invalid record in {table_name}: {row}")
            continue

        # Only use columns that actually exist in the database.
        columns = [
            column for column in row.keys()
            if column in existing_columns
        ]

        if not columns:
            print(f"WARNING: No matching columns for {table_name}")
            continue

        values = [row[column] for column in columns]

        column_sql = ", ".join(f'"{column}"' for column in columns)
        placeholders = ", ".join("?" for _ in columns)

        sql = f"""
            INSERT OR IGNORE INTO "{table_name}"
            ({column_sql})
            VALUES ({placeholders})
        """

        before = cursor.rowcount

        cursor.execute(sql, values)

        # SQLite rowcount is 1 when inserted and 0 when ignored.
        if cursor.rowcount == 1:
            inserted += 1
        else:
            skipped += 1

    print(
        f"{table_name}: "
        f"{inserted} inserted, "
        f"{skipped} already existed"
    )

    return inserted


def main():
    print("=" * 60)
    print("JAL-SETU JSON TEST DATA IMPORT")
    print("=" * 60)

    if not os.path.exists(DB_FILE):
        print(f"ERROR: Database '{DB_FILE}' was not found.")
        sys.exit(1)

    data = load_json()

    if not isinstance(data, dict):
        print("ERROR: JSON root must be an object.")
        sys.exit(1)

    print(f"\nDatabase: {DB_FILE}")
    print(f"JSON file: {JSON_FILE}\n")

    connection = sqlite3.connect(DB_FILE)

    # Make SQLite enforce foreign-key relationships.
    connection.execute("PRAGMA foreign_keys = ON")

    cursor = connection.cursor()

    try:
        print("Checking existing database tables...\n")

        for table in IMPORT_ORDER:
            if table_exists(cursor, table):
                print(f"✓ {table}")
            else:
                print(f"✗ {table} NOT FOUND")

        print("\nStarting import...\n")

        total_inserted = 0

        # Everything happens inside one transaction.
        for table in IMPORT_ORDER:
            rows = data.get(table, [])

            if not isinstance(rows, list):
                print(
                    f"WARNING: '{table}' should contain "
                    f"a JSON array. Skipping."
                )
                continue

            total_inserted += import_table(
                cursor,
                table,
                rows
            )

        # Check foreign-key consistency before committing.
        foreign_key_errors = cursor.execute(
            "PRAGMA foreign_key_check"
        ).fetchall()

        if foreign_key_errors:
            print("\nERROR: Foreign-key problems detected:")
            for error in foreign_key_errors:
                print(error)

            print("\nRolling back import...")
            connection.rollback()
            sys.exit(1)

        # Everything is valid.
        connection.commit()

        print("\n" + "=" * 60)
        print("IMPORT SUCCESSFUL")
        print("=" * 60)
        print(f"New records inserted: {total_inserted}")
        print("Existing records were NOT overwritten.")
        print("Foreign-key relationships are valid.")

    except Exception as e:
        print("\nERROR during import:")
        print(e)

        print("\nRolling back...")
        connection.rollback()

        print("No changes from this import were saved.")
        sys.exit(1)

    finally:
        connection.close()


if __name__ == "__main__":
    main()