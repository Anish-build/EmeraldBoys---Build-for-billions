from database.database import get_connection
from datetime import datetime
import uuid


def create_pump(pump_id, location, pump_type=None, operator=None):

    conn = get_connection()
    cursor = conn.cursor()

    try:
        now = datetime.now().isoformat()

        cursor.execute("""
            INSERT INTO pumps (
                pump_id,
                location,
                pump_type,
                operator,
                status,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            pump_id,
            location,
            pump_type,
            operator,
            "Active",
            now,
            now
        ))

        conn.commit()

        return pump_id

    finally:
        conn.close()


def get_pump(pump_id):

    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT *
            FROM pumps
            WHERE pump_id = ?
        """, (pump_id,))

        return cursor.fetchone()

    finally:
        conn.close()


def create_inspection(pump_id, audio_file):

    inspection_id = "I-" + str(uuid.uuid4())[:8]
    now = datetime.now().isoformat()

    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            INSERT INTO inspections (
                inspection_id,
                pump_id,
                audio_file,
                timestamp
            )
            VALUES (?, ?, ?, ?)
        """, (
            inspection_id,
            pump_id,
            audio_file,
            now
        ))

        conn.commit()

        return inspection_id

    finally:
        conn.close()


def save_diagnosis(
    inspection_id,
    prediction,
    confidence,
    severity
):

    diagnosis_id = "D-" + str(uuid.uuid4())[:8]
    now = datetime.now().isoformat()

    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            INSERT INTO diagnoses (
                diagnosis_id,
                inspection_id,
                prediction,
                confidence,
                severity,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            diagnosis_id,
            inspection_id,
            prediction,
            confidence,
            severity,
            now
        ))

        conn.commit()

        return diagnosis_id

    finally:
        conn.close()


def create_mechanic(
    name,
    skill,
    service_area
):

    mechanic_id = "M-" + str(uuid.uuid4())[:8]

    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            INSERT INTO mechanics (
                mechanic_id,
                name,
                skill,
                service_area,
                available,
                workload
            )
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            mechanic_id,
            name,
            skill,
            service_area,
            1,
            0
        ))

        conn.commit()

        return mechanic_id

    finally:
        conn.close()


def create_repair(
    pump_id,
    inspection_id,
    mechanic_id,
    recommended_action,
    priority
):

    repair_id = "R-" + str(uuid.uuid4())[:8]
    now = datetime.now().isoformat()

    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            INSERT INTO repairs (
                repair_id,
                pump_id,
                inspection_id,
                mechanic_id,
                recommended_action,
                priority,
                status,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            repair_id,
            pump_id,
            inspection_id,
            mechanic_id,
            recommended_action,
            priority,
            "Dispatched",
            now
        ))

        conn.commit()

        return repair_id

    finally:
        conn.close()


def save_verification(
    repair_id,
    before_prediction,
    after_prediction,
    before_confidence,
    after_confidence,
    result
):

    verification_id = "V-" + str(uuid.uuid4())[:8]
    now = datetime.now().isoformat()

    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            INSERT INTO verifications (
                verification_id,
                repair_id,
                before_prediction,
                after_prediction,
                before_confidence,
                after_confidence,
                result,
                verified_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            verification_id,
            repair_id,
            before_prediction,
            after_prediction,
            before_confidence,
            after_confidence,
            result,
            now
        ))

        conn.commit()

        return verification_id

    finally:
        conn.close()


def get_available_mechanics():

    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT *
            FROM mechanics
            WHERE available = 1
            ORDER BY workload ASC
        """)

        return cursor.fetchall()

    finally:
        conn.close()


def get_pump_history(pump_id):

    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT
                i.inspection_id,
                i.audio_file,
                i.timestamp,
                d.diagnosis_id,
                d.prediction,
                d.confidence,
                d.severity,
                r.repair_id,
                r.recommended_action,
                r.priority,
                r.status
            FROM inspections i

            LEFT JOIN diagnoses d
                ON i.inspection_id = d.inspection_id

            LEFT JOIN repairs r
                ON i.inspection_id = r.inspection_id

            WHERE i.pump_id = ?

            ORDER BY i.timestamp DESC
        """, (pump_id,))

        return cursor.fetchall()

    finally:
        conn.close()


def update_repair_status(
    repair_id,
    status
):

    conn = get_connection()
    cursor = conn.cursor()

    try:
        completed_at = None

        if status == "Completed":
            completed_at = datetime.now().isoformat()

        cursor.execute("""
            UPDATE repairs
            SET status = ?,
                completed_at = ?
            WHERE repair_id = ?
        """, (
            status,
            completed_at,
            repair_id
        ))

        conn.commit()

    finally:
        conn.close()


def log_event(
    pump_id,
    inspection_id,
    event_type,
    action,
    reason
):

    log_id = "L-" + str(uuid.uuid4())[:8]
    now = datetime.now().isoformat()

    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            INSERT INTO audit_log (
                log_id,
                pump_id,
                inspection_id,
                event_type,
                action,
                reason,
                timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            log_id,
            pump_id,
            inspection_id,
            event_type,
            action,
            reason,
            now
        ))

        conn.commit()

        return log_id

    finally:
        conn.close()


def get_latest_diagnosis(pump_id):

    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT
                i.inspection_id,
                i.audio_file,
                i.timestamp,
                d.diagnosis_id,
                d.prediction,
                d.confidence,
                d.severity
            FROM inspections i

            JOIN diagnoses d
                ON i.inspection_id = d.inspection_id

            WHERE i.pump_id = ?

            ORDER BY i.timestamp DESC

            LIMIT 1
        """, (pump_id,))

        return cursor.fetchone()

    finally:
        conn.close()


def get_latest_repair(pump_id):

    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT
                r.repair_id,
                r.inspection_id,
                r.mechanic_id,
                r.recommended_action,
                r.priority,
                r.status,
                r.created_at,
                r.completed_at
            FROM repairs r

            WHERE r.pump_id = ?

            ORDER BY r.created_at DESC

            LIMIT 1
        """, (pump_id,))

        return cursor.fetchone()

    finally:
        conn.close()


def get_latest_verification(pump_id):

    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT
                v.verification_id,
                v.repair_id,
                v.before_prediction,
                v.after_prediction,
                v.before_confidence,
                v.after_confidence,
                v.result,
                v.verified_at
            FROM verifications v

            JOIN repairs r
                ON v.repair_id = r.repair_id

            WHERE r.pump_id = ?

            ORDER BY v.verified_at DESC

            LIMIT 1
        """, (pump_id,))

        return cursor.fetchone()

    finally:
        conn.close()


def get_pump_status(pump_id):

    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT
                p.pump_id,
                p.location,
                p.status,

                d.prediction,
                d.confidence,
                d.severity,

                r.repair_id,
                r.priority,
                r.status,

                v.result

            FROM pumps p

            LEFT JOIN inspections i
                ON p.pump_id = i.pump_id

            LEFT JOIN diagnoses d
                ON i.inspection_id = d.inspection_id

            LEFT JOIN repairs r
                ON i.inspection_id = r.inspection_id

            LEFT JOIN verifications v
                ON r.repair_id = v.repair_id

            WHERE p.pump_id = ?

            ORDER BY i.timestamp DESC

            LIMIT 1
        """, (pump_id,))

        return cursor.fetchone()

    finally:
        conn.close()