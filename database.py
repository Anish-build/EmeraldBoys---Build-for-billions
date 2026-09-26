import os
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
    func
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

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
    """Stores handpump metadata and current lifecycle status."""
    __tablename__ = "pumps"

    id = Column(Integer, primary_key=True, index=True)
    pump_id = Column(String(50), unique=True, index=True, nullable=False)
    status = Column(String(50), default="DIAGNOSIS_PENDING", nullable=False)
    location_info = Column(Text, nullable=True)
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


def init_db() -> None:
    """Creates SQLite directory and initializes tables safely if not already present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Dependency helper to yield database session and ensure clean close."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
