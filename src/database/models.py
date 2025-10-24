# Database models for survey persistence
# Using SQLAlchemy ORM for PostgreSQL integration

from sqlalchemy import Column, String, DateTime, JSON, Enum as SQLEnum, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from src.api.models.surveys import SurveyStatus

Base = declarative_base()

class Survey(Base):
    """Survey database model for persistent storage"""
    __tablename__ = 'surveys'
    
    survey_id = Column(String, primary_key=True)
    product_name = Column(String, nullable=False)
    product_description = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(SQLEnum(SurveyStatus), default=SurveyStatus.PENDING)
    configuration = Column(JSON, nullable=False)
    survey_metadata = Column(JSON)
    
    def to_dict(self):
        """Convert database model to dictionary"""
        return {
            'survey_id': self.survey_id,
            'product_name': self.product_name,
            'product_description': self.product_description,
            'created_at': self.created_at,
            'status': self.status,
            'configuration': self.configuration,
            'metadata': self.survey_metadata
        }

# Database connection setup
DATABASE_URL = os.environ.get('DATABASE_URL')
if DATABASE_URL:
    # SQLAlchemy 2.0 requires postgresql:// to be changed to postgresql+psycopg2://
    if DATABASE_URL.startswith('postgresql://'):
        DATABASE_URL = DATABASE_URL.replace('postgresql://', 'postgresql+psycopg2://', 1)
    
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_recycle=300,
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Create tables
    Base.metadata.create_all(bind=engine)
else:
    engine = None
    SessionLocal = None

def get_db():
    """Get database session"""
    if SessionLocal:
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()
    else:
        yield None