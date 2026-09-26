from database.database import get_connection


def create_tables():

    conn = get_connection()
    cursor = conn.cursor()

    # 1. Pumps table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pumps (
            pump_id TEXT PRIMARY KEY,
            location TEXT NOT NULL,
            pump_type TEXT,
            operator TEXT,
            status TEXT DEFAULT 'Active',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    # 2. Inspections table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS inspections (
            inspection_id TEXT PRIMARY KEY,
            pump_id TEXT NOT NULL,
            audio_file TEXT,
            timestamp TEXT NOT NULL,

            FOREIGN KEY (pump_id)
                REFERENCES pumps(pump_id)
        )
    """)

    # 3. Diagnoses table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS diagnoses (
            diagnosis_id TEXT PRIMARY KEY,
            inspection_id TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            severity TEXT,
            created_at TEXT NOT NULL,

            FOREIGN KEY (inspection_id)
                REFERENCES inspections(inspection_id)
        )
    """)

    # 4. Mechanics table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS mechanics (
            mechanic_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            skill TEXT,
            service_area TEXT,
            available INTEGER DEFAULT 1,
            workload INTEGER DEFAULT 0
        )
    """)

    # 5. Repairs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS repairs (
            repair_id TEXT PRIMARY KEY,
            pump_id TEXT NOT NULL,
            inspection_id TEXT NOT NULL,
            mechanic_id TEXT,
            recommended_action TEXT,
            priority TEXT,
            status TEXT DEFAULT 'Dispatched',
            created_at TEXT NOT NULL,
            completed_at TEXT,

            FOREIGN KEY (pump_id)
                REFERENCES pumps(pump_id),

            FOREIGN KEY (inspection_id)
                REFERENCES inspections(inspection_id),

            FOREIGN KEY (mechanic_id)
                REFERENCES mechanics(mechanic_id)
        )
    """)

    # 6. Verifications table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS verifications (
            verification_id TEXT PRIMARY KEY,
            repair_id TEXT NOT NULL,
            before_prediction TEXT,
            after_prediction TEXT,
            before_confidence REAL,
            after_confidence REAL,
            result TEXT NOT NULL,
            verified_at TEXT NOT NULL,

            FOREIGN KEY (repair_id)
                REFERENCES repairs(repair_id)
        )
    """)

    # 7. Audit log table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            log_id TEXT PRIMARY KEY,
            pump_id TEXT,
            inspection_id TEXT,
            event_type TEXT,
            action TEXT,
            reason TEXT,
            timestamp TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()

    print("Database tables created successfully!")


if __name__ == "__main__":
    create_tables()