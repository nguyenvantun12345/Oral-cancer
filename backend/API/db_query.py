import logging
from typing import List, Optional
from bson import ObjectId
from pymongo import MongoClient
from marshmallow import ValidationError
from db_schema import PatientSchema, MedicalHistorySchema, get_collections
from auth_utils import hash_password
from datetime import datetime

logger = logging.getLogger(__name__)

class Patient:
    def __init__(self):
        self.collections = get_collections()
        self.collection = self.collections["User"]
        self.schema = PatientSchema()
        self.medical_crud = MedicalHistoryCRUD()

    def create_many(self, data_list: List[dict]) -> List[str]:
        """Tạo nhiều bệnh nhân trong MongoDB."""
        try:
            validated_data = [self.schema.load(data) for data in data_list]
            # for doc in validated_data:
            #     if 'password' in doc:
            #         doc['password'] = hash_password(doc['password'])
            result = self.collection.insert_many(validated_data)
            logger.info(f"Inserted {len(result.inserted_ids)} documents")
            return [str(doc["user_id"]) for doc in validated_data]
        except ValidationError as e:
            logger.error(f"Validation error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error creating documents: {str(e)}")
            raise

    def createPatient(self, patients: List[dict]) -> List[str]:
        """Tạo bệnh nhân (gọi create_many)."""
        return self.create_many(patients)

    def get_patient_by_id(self, user_id: str) -> Optional[dict]:
        """Lấy thông tin bệnh nhân theo user_id."""
        try:
            patient = self.collection.find_one({"user_id": user_id})
            if patient:
                logger.info(f"Found patient with user_id: {user_id}")
                return patient
            else:
                logger.warning(f"No patient found with user_id: {user_id}")
                return None
        except Exception as e:
            logger.error(f"Error retrieving patient with user_id {user_id}: {str(e)}")
            return None

    def search_patients(self, search_text: str, limit: int = 10, skip: int = 0) -> List[dict]:
        """Tìm kiếm bệnh nhân theo tên hoặc username."""
        try:
            query = {"$or": [
                {"name": {"$regex": search_text, "$options": "i"}},
                {"username": {"$regex": search_text, "$options": "i"}}
            ]} if search_text else {}
            patients = list(self.collection.find(query).skip(skip).limit(limit))
            logger.info(f"Found {len(patients)} patients matching search_text: {search_text}")
            return patients
        except Exception as e:
            logger.error(f"Error searching patients: {str(e)}")
            return []

    def update_patient(self, user_id: str, patient_data: dict) -> bool:
        """Cập nhật thông tin bệnh nhân và thêm thời gian vào mảng date trong MedicalHistory nếu tồn tại."""
        try:
            validated_data = self.schema.load(patient_data, partial=True)
            result = self.collection.update_one(
                {"user_id": user_id},
                {"$set": validated_data}
            )
            if result.modified_count == 0 and result.matched_count == 0:
                logger.warning(f"No patient found or updated for user_id: {user_id}")
                return False
            
            # Chỉ cập nhật MedicalHistory nếu đã tồn tại
            existing_history = self.medical_crud.get_medical_history_by_user_id(user_id)
            if existing_history:
                medical_update = {
                    "user_id": user_id,
                    "date": existing_history.get("date", []) + [datetime.now().isoformat()]
                }
                self.medical_crud.update_medical_history(user_id, medical_update)
                logger.info(f"Updated medical history date for user_id: {user_id}")
            
            logger.info(f"Updated patient for user_id: {user_id}")
            return True
        except ValidationError as e:
            logger.error(f"Validation error: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error updating patient: {str(e)}")
            return False

    def delete_patient(self, user_id: str) -> bool:
        """Xóa bệnh nhân theo user_id."""
        try:
            result = self.collection.delete_one({"user_id": user_id})
            if result.deleted_count > 0:
                self.medical_crud.delete_medical_history(user_id)
                logger.info(f"Deleted patient with user_id: {user_id}")
                return True
            logger.warning(f"No patient found to delete with user_id: {user_id}")
            return False
        except Exception as e:
            logger.error(f"Error deleting patient with user_id {user_id}: {str(e)}")
            return False

class MedicalHistoryCRUD:
    def __init__(self):
        self.collections = get_collections()
        self.collection = self.collections["MedicalHistory"]
        self.schema = MedicalHistorySchema()

    def create_medical_history(self, medical_history: dict) -> bool:
        """Create a new medical history for a user (multiple allowed per user)."""
        try:
            validated_data = self.schema.load(medical_history)
            validated_data["date"] = validated_data.get("date", []) + [datetime.now().isoformat()]
            result = self.collection.insert_one(validated_data)
            if result.inserted_id:
                logger.info(f"Created medical history with image_id: {validated_data['image_id']}")
                return True
            return False
        except ValidationError as e:
            logger.error(f"Validation error for medical history: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error creating medical history: {str(e)}")
            return False

    def update_medical_history(self, user_id: str, medical_data: dict) -> bool:
        """Cập nhật lịch sử y tế, thêm thời gian vào mảng date."""
        try:
            validated_data = self.schema.load(medical_data, partial=True)
            current_history = self.get_medical_history_by_user_id(user_id)
            
            validated_data["date"] = current_history.get("date", []) + [datetime.now().isoformat()]
            
            result = self.collection.update_one(
                {"user_id": user_id},
                {"$set": validated_data},
                upsert=True
            )
            success = result.modified_count > 0 or result.upserted_id is not None
            if success:
                logger.info(f"Updated medical history with new date for user_id: {user_id}")
            else:
                logger.warning(f"No changes made to medical history for user_id: {user_id}")
            return success
        except ValidationError as e:
            logger.error(f"Validation error for medical history: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error updating medical history: {str(e)}")
            return False

    def get_medical_history_by_user_id(self, user_id: str) -> list:
        """Get all medical histories for a user as a list."""
        try:
            histories = list(self.collection.find({"user_id": user_id}))
            # Convert ObjectId to str for each document
            for h in histories:
                if '_id' in h:
                    h['_id'] = str(h['_id'])
                if 'date' in h and isinstance(h['date'], list):
                    h['date'] = [d.isoformat() if hasattr(d, 'isoformat') else d for d in h['date']]
            return histories
        except Exception as e:
            logger.error(f"Error retrieving medical histories for user_id {user_id}: {str(e)}")
            return []

    def delete_medical_history(self, user_id: str) -> bool:
        """Xóa lịch sử y tế theo user_id."""
        try:
            result = self.collection.delete_one({"user_id": user_id})
            if result.deleted_count > 0:
                logger.info(f"Deleted medical history for user_id: {user_id}")
                return True
            logger.warning(f"No medical history found to delete for user_id: {user_id}")
            return False
        except Exception as e:
            logger.error(f"Error deleting medical history for user_id {user_id}: {str(e)}")
            return False