from pydantic import BaseModel, Field, validator
from typing import Dict, Optional
from datetime import datetime, timedelta, timezone
from core.checkquailty import load_image
import logging

logger = logging.getLogger(__name__)


class PatientCreate(BaseModel):
    name: Optional[str]
    birthdate: Optional[str]
    gender: Optional[str]
    role: Optional[str]
    work: Optional[str]
    username: str
    email: Optional[str]
    password: str

    @validator('birthdate', pre=True, always=True)
    def validate_birthdate(cls, value):
        if value:
            try:
                datetime.strptime(value, '%d/%m/%Y')
            except ValueError:
                raise ValueError("birthdate must be in format dd/mm/yyyy")
        return value

class PatientUpdate(BaseModel):
    name: Optional[str]
    birthdate: Optional[str] = Field(None, pattern=r'^\d{2}/\d{2}/\d{4}$')
    gender: Optional[str]
    role: Optional[str]
    work: Optional[str]
    email: Optional[str]

    @validator('birthdate', pre=True, always=True)
    def validate_birthdate(cls, value):
        if value:
            try:
                datetime.strptime(value, '%d/%m/%Y')
            except ValueError:
                raise ValueError("birthdate must be in format dd/mm/yyyy")
        return value

class ImageCreate(BaseModel):
    image_id: str
    image: str  # URL or local path
    diagnosis_score: float
    comment: Optional[str]

    @validator('image')
    def validate_image(cls, value):
        logger.info(f"Validating image: {value}")
        try:
            img = load_image(value)
            if img is None:
                raise ValueError(f"Image cannot be loaded from {value}. Please provide a valid URL or file path.")
            return value
        except Exception as e:
            logger.error(f"Failed to validate image {value}: {str(e)}")
            raise ValueError(f"Invalid image: {str(e)}")

class ImageUpdate(BaseModel):
    image: Optional[str]
    diagnosis_score: Optional[float]
    comment: Optional[str]

    @validator('image', pre=True, always=True)
    def validate_image(cls, value):
        if value:
            logger.info(f"Validating image: {value}")
            try:
                img = load_image(value)
                if img is None:
                    raise ValueError(f"Image cannot be loaded from {value}. Please provide a valid URL or file path.")
                return value
            except Exception as e:
                logger.error(f"Failed to validate image {value}: {str(e)}")
                raise ValueError(f"Invalid image: {str(e)}")
        return value
