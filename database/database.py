import sqlite3


DATABASE = "jalsetu.db"


def get_connection():
    connection = sqlite3.connect(
        DATABASE,
        timeout=10
    )

    connection.execute("PRAGMA foreign_keys = ON")

    return connection