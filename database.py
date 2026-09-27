import os
import logging
from datetime import datetime, timezone
from typing import Generator
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    Boolean,
    Text,
    DateTime,
    ForeignKey,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

logger = logging.getLogger("EmeraldDatabase")

# Define paths and ensure data directory exists
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATABASE_PATH = os.path.join(DATA_DIR, "emerald_boys.db")
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

# Initialize Engine with multithreaded support for SQLite
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Pump(Base):
    """Stores handpump metadata, hardware context, and authoritative lifecycle status."""
    __tablename__ = "pumps"

    id = Column(Integer, primary_key=True, index=True)
    pump_id = Column(String(50), unique=True, index=True, nullable=False)
    status = Column(String(50), default="DIAGNOSIS_PENDING", nullable=False)
    location_info = Column(Text, nullable=True)
    age_years = Column(Integer, default=5, nullable=False)
    last_maintenance_date = Column(String(50), default="2023-01-15", nullable=True)
    previous_failures = Column(Integer, default=2, nullable=False)
    known_issues = Column(Text, default="None", nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False
    )


class DiagnosticCase(Base):
    """Stores the ML input, agent decision, and deterministic validation result."""
    __tablename__ = "diagnostic_cases"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(String(50), index=True, nullable=False)
    pump_id = Column(String(50), ForeignKey("pumps.pump_id"), index=True, nullable=False)
    ml_prediction = Column(String(50), nullable=False)
    ml_confidence = Column(Float, nullable=False)
    confidence_flag = Column(String(20), nullable=False)
    input_state = Column(String(50), nullable=False)
    proposed_state = Column(String(50), nullable=False)
    final_state = Column(String(50), nullable=False)
    action = Column(String(50), nullable=False)
    needs_human_review = Column(Boolean, default=False, nullable=False)
    explanation = Column(Text, nullable=False)
    audit_notes = Column(Text, nullable=False)
    was_overridden = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False
    )


class StateTransition(Base):
    """Stores full historical audit trail of state transitions and overrides."""
    __tablename__ = "state_transitions"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(String(50), index=True, nullable=False)
    pump_id = Column(String(50), ForeignKey("pumps.pump_id"), index=True, nullable=False)
    from_state = Column(String(50), nullable=False)
    proposed_state = Column(String(50), nullable=False)
    final_state = Column(String(50), nullable=False)
    action = Column(String(50), nullable=False)
    reason = Column(Text, nullable=True)
    was_overridden = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)


def seed_default_pumps(db: Session) -> None:
    """Seeds baseline pump records and physical contexts if not already present."""
    default_pumps = [
        {
            "pump_id": "PUMP-001",
            "status": "DIAGNOSIS_PENDING",
            "location_info": "Kibera Cluster 4, Well Station A",
            "age_years": 8,
            "last_maintenance_date": "2023-01-15",
            "previous_failures": 3,
            "known_issues": "Prone to severe seal degradation."
        },
        {
            "pump_id": "PUMP-002",
            "status": "DIAGNOSIS_PENDING",
            "location_info": "Turkana Well 2, North Sub-County",
            "age_years": 1,
            "last_maintenance_date": "2023-09-01",
            "previous_failures": 0,
            "known_issues": "None"
        },
        {
            "pump_id": "PUMP-003",
            "status": "HEALTHY",
            "location_info": "Marsabit Station 1, Central Basin",
            "age_years": 4,
            "last_maintenance_date": "2024-02-10",
            "previous_failures": 1,
            "known_issues": "Bearing vibration detected in Q1."
        }
    ]

    for p_data in default_pumps:
        existing = db.query(Pump).filter(Pump.pump_id == p_data["pump_id"]).first()
        if not existing:
            pump = Pump(**p_data)
            db.add(pump)
            logger.info(f"Seeded default pump: {p_data['pump_id']}")
    try:
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to seed default pumps: {e}")


def init_db() -> None:
    """Creates SQLite directory, initializes tables, and seeds initial data."""
    os.makedirs(DATA_DIR, exist_ok=True)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        seed_default_pumps(db)
    finally:
        db.close()


def get_db() -> Generator[Session, None, None]:
    """Dependency helper to yield database session and ensure clean close."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
