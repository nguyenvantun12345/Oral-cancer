import logging
from datetime import datetime
from bson import ObjectId
from marshmallow import Schema, fields, validate, ValidationError, pre_load
from pymongo import MongoClient, TEXT
from db_config import get_database
logger = logging.getLogger(__name__)

def get_collections():
    db = get_database()
    return {
        "User": db["User"],
        "MedicalHistory": db["MedicalHistory"]
    }

class MedicalHistorySchema(Schema):
    image_id = fields.String(load_default=lambda: str(ObjectId()))
    user_id = fields.String(required=True)
    image = fields.String(required=True, validate=validate.URL())
    comment = fields.String(allow_none=True)
    date = fields.List(fields.DateTime(), load_default=lambda: [datetime.utcnow()])
    diagnosis_score = fields.Float(allow_none=True, validate=validate.Range(min=0, max=1))
    quality_report = fields.Dict(allow_none=True)  # Added to handle quality_report field

class PatientSchema(Schema):
    user_id = fields.String(load_default=lambda: str(ObjectId()))
    name = fields.String(required=True, validate=validate.Length(min=1, max=200))
    role = fields.String(required=True, validate=validate.OneOf(['patient', 'doctor', 'admin']))
    email = fields.Email(required=True)
    gender = fields.String(required=True, validate=validate.OneOf(['male', 'female', 'other']))
    phone = fields.String(allow_none=True, validate=validate.Length(max=30))
    work = fields.String(allow_none=True, validate=validate.Length(max=500))
    birthdate = fields.String(required=True)
    username = fields.String(required=True, validate=validate.Length(min=3, max=60))
    password = fields.String(required=True, load_only=True)

    @pre_load
    def parse_birthdate(self, data, **kwargs):
        """Chuyển đổi birthdate từ dd/mm/yyyy sang định dạng YYYY-MM-DD"""
        if 'birthdate' in data and isinstance(data['birthdate'], str):
            try:
                parsed_date = datetime.strptime(data['birthdate'], '%d/%m/%Y')
                data['birthdate'] = parsed_date.strftime('%Y-%m-%d')
            except ValueError:
                raise ValidationError("birthdate must be in format dd/mm/yyyy", field_name="birthdate")
        return data

def create_indexes():
    try:
        collections = get_collections()
        user_collection = collections["User"]
        medical_history_collection = collections["MedicalHistory"]

        # Chỉ mục cho User
        user_collection.create_index([("user_id", 1)], unique=True)
        user_collection.create_index([("name", TEXT), ("username", TEXT)])
        user_collection.create_index([("email", 1)], unique=True)
        user_collection.create_index([("username", 1)], unique=True)

        # Chỉ mục cho MedicalHistory
        medical_history_collection.create_index([("image_id", 1)], unique=True)
        medical_history_collection.create_index([("user_id", 1)], unique=True)

        logger.info("Created indexes for User and MedicalHistory collections")
    except Exception as e:
        logger.error(f"Error creating indexes: {str(e)}")
        raise