import uuid
from datetime import datetime
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    Text,
    JSON,
)
from sqlalchemy.orm import relationship
from app.core.database import Base


def generate_uuid():
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    datasets = relationship("Dataset", back_populates="user", cascade="all, delete-orphan")


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    owner_token = Column(String(64), nullable=True, index=True)
    filename = Column(String(255), nullable=False)
    file_size_bytes = Column(Integer, default=0)
    raw_file_path = Column(String(512), nullable=False)
    cleaned_file_path = Column(String(512), nullable=True)
    row_count = Column(Integer, default=0)
    col_count = Column(Integer, default=0)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="datasets")
    objective = relationship("Objective", back_populates="dataset", uselist=False, cascade="all, delete-orphan")
    profile_report = relationship("ProfileReport", back_populates="dataset", uselist=False, cascade="all, delete-orphan")
    issues = relationship("IssueDetection", back_populates="dataset", cascade="all, delete-orphan")
    health_scores = relationship("HealthScore", back_populates="dataset", cascade="all, delete-orphan")
    recommendations = relationship("Recommendation", back_populates="dataset", cascade="all, delete-orphan")
    processing_jobs = relationship("ProcessingJob", back_populates="dataset", cascade="all, delete-orphan")
    benchmarks = relationship("ModelBenchmark", back_populates="dataset", cascade="all, delete-orphan")
    snapshot = relationship("DatasetSnapshot", back_populates="dataset", uselist=False, cascade="all, delete-orphan")


class DatasetSnapshot(Base):
    """Last API payloads per stage, used to restore the UI session after a page refresh."""
    __tablename__ = "dataset_snapshots"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=False, unique=True)
    diagnostics = Column(JSON, nullable=True)
    execution = Column(JSON, nullable=True)
    leaderboard = Column(JSON, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    dataset = relationship("Dataset", back_populates="snapshot")


class Objective(Base):
    __tablename__ = "objectives"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=False, unique=True)
    problem_type = Column(String(50), nullable=False)  # "classification" | "regression"
    target_column = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    dataset = relationship("Dataset", back_populates="objective")


class ProfileReport(Base):
    __tablename__ = "profile_reports"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=False, unique=True)
    column_profiles = Column(JSON, nullable=False)  # list of dicts
    summary_stats = Column(JSON, nullable=False)    # dataset level stats
    created_at = Column(DateTime, default=datetime.utcnow)

    dataset = relationship("Dataset", back_populates="profile_report")


class IssueDetection(Base):
    __tablename__ = "issue_detections"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=False)
    issue_type = Column(String(100), nullable=False)
    column = Column(String(255), nullable=True)
    severity = Column(String(20), default="medium")  # high, medium, low, info
    details = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    dataset = relationship("Dataset", back_populates="issues")


class HealthScore(Base):
    __tablename__ = "health_scores"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=False)
    stage = Column(String(20), nullable=False)  # "before" | "after"
    composite_score = Column(Float, nullable=False)  # 0 to 100
    sub_scores = Column(JSON, nullable=False)        # dict of sub-scores
    summary_text = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    dataset = relationship("Dataset", back_populates="health_scores")


class Recommendation(Base):
    __tablename__ = "recommendations"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=False)
    issue_type = Column(String(100), nullable=False)
    column = Column(String(255), nullable=True)
    method = Column(String(100), nullable=False)
    reason_title = Column(String(255), nullable=False)
    explanation_text = Column(Text, nullable=False)
    explanation_source = Column(String(20), nullable=True)  # "ai" | "statistical"
    params = Column(JSON, nullable=True)
    severity = Column(String(20), default="medium")
    is_destructive = Column(Boolean, default=False)
    is_approved = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    dataset = relationship("Dataset", back_populates="recommendations")


class ProcessingJob(Base):
    __tablename__ = "processing_jobs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=False)
    status = Column(String(50), default="pending")  # pending, running, completed, failed
    error_message = Column(Text, nullable=True)
    pipeline_file_path = Column(String(512), nullable=True)
    script_file_path = Column(String(512), nullable=True)
    report_file_path = Column(String(512), nullable=True)
    applied_steps = Column(JSON, nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    dataset = relationship("Dataset", back_populates="processing_jobs")


class ModelBenchmark(Base):
    __tablename__ = "model_benchmarks"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=False)
    model_name = Column(String(100), nullable=False)
    problem_type = Column(String(50), nullable=False)
    metric_name = Column(String(50), nullable=False)
    metric_value = Column(Float, nullable=False)
    training_time_sec = Column(Float, default=0.0)
    rank = Column(Integer, default=1)
    status = Column(String(20), default="success")
    is_recommended = Column(Boolean, default=False)
    details = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    dataset = relationship("Dataset", back_populates="benchmarks")
