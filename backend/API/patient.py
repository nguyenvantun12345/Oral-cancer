from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, validator
from typing import Dict, Optional, List
from datetime import datetime, timezone
import logging
import requests
from db_redis import RedisCache
from api import get_current_user, PatientUpdate, validate_age
from db_schema import PatientSchema, MedicalHistorySchema
from marshmallow import ValidationError
from pydantic import BaseModel
from diagnosis import predict_from_url

class MedicalHistoryResponse(BaseModel):
    _id: Optional[str]
    image_id: str
    user_id: str
    image: str
    comment: Optional[str]
    date: List[str]
    diagnosis_score: Optional[float]

class MedicalHistoryListResponse(BaseModel):
    histories: List[MedicalHistoryResponse]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/patients", tags=["Patient"])

class ImageCreatePatient(BaseModel):
    image: str

    @validator('image')
    def validate_image(cls, value):
        from urllib.parse import urlparse
        parsed = urlparse(value)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("Image must be a valid URL")
        return value

class ImageUpdatePatient(BaseModel):
    image: Optional[str]

    @validator('image', pre=True, always=True)
    def validate_image(cls, value):
        if value:
            from urllib.parse import urlparse
            parsed = urlparse(value)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError("Image must be a valid URL")
        return value

@router.put("/me", response_model=Dict)
async def update_current_patient(patient: PatientUpdate, current_user: Dict = Depends(get_current_user)):
    try:
        if current_user['role'] != 'patient':
            raise HTTPException(status_code=403, detail="Only patients can update their own data")

        patient_data = patient.dict(exclude_unset=True)
        patient_schema = PatientSchema()
        try:
            patient_schema.load(patient_data, partial=True)
        except ValidationError as ve:
            raise HTTPException(status_code=400, detail=f"Invalid input: {ve.messages}. Please correct the data and try again.")

        if 'birthdate' in patient_data and patient_data['birthdate']:
            is_valid, error_message = validate_age(patient_data['birthdate'])
            if not is_valid:
                raise HTTPException(status_code=400, detail=f"Invalid birthdate: {error_message}. Please correct and try again.")

        redis_cache = RedisCache()
        if not redis_cache.update_patient(current_user['user_id'], patient_data):
            raise HTTPException(status_code=404, detail="Patient not found")

        updated_patient = redis_cache.get_patient_by_id(current_user['user_id'])
        redis_cache.log_audit(
            action="update_patient",
            user_id=current_user['user_id'],
            details={"user_id": current_user['user_id']}
        )

        logger.info(f"Updated patient: {current_user['user_id']}")
        return updated_patient
    except HTTPException as he:
        logger.error(f"HTTP error updating patient: {he.detail}")
        raise
    except Exception as e:
        logger.error(f"Error updating patient: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/me/history", response_model=MedicalHistoryListResponse)
async def create_current_medical_history(image: ImageCreatePatient, 
                                        current_user: Dict = Depends(get_current_user)):
    try:
        if current_user['role'] != 'patient':
            raise HTTPException(status_code=403, detail="Only patients can manage their own data")

        image_data = image.dict()
        image_data['user_id'] = current_user['user_id']
        image_data['comment'] = None
        image_data['diagnosis_score'] = None

        # --- Integrate diagnosis model call ---
        try:
            # Local function call instead of API call
            diagnosis_result = predict_from_url(image_data['image'])
            print(diagnosis_result)
            # Extract confidence score
            diagnosis_score = diagnosis_result.get('probabilities', {}).get('Cancer')
            if diagnosis_score is None:
                raise Exception("Prediction result did not return a confidence score.")

            # Add score to image data
            image_data['diagnosis_score'] = diagnosis_score

        except Exception as diag_e:
            logger.error(f"Diagnosis failed: {diag_e}")
            raise HTTPException(status_code=500, detail=f"Diagnosis failed: {diag_e}")
        medical_schema = MedicalHistorySchema()
        try:
            validated_data = medical_schema.load(image_data)
        except ValidationError as ve:
            raise HTTPException(status_code=400, detail=f"Invalid medical history data: {ve.messages}. Please correct the data and try again.")

        redis_cache = RedisCache()
        validated_data['date'] = [datetime.now(timezone.utc).isoformat()]
        if not redis_cache.create_medical_history(validated_data):
            raise HTTPException(status_code=400, detail="Failed to add medical history")

        redis_cache.log_audit(
            action="create_medical_history",
            user_id=current_user['user_id'],
            details={"user_id": current_user['user_id'], "image_id": validated_data['image_id']}
        )

        logger.info(f"Created medical history {validated_data['image_id']} for patient {current_user['user_id']}")
        # Return the latest created history (with diagnosis_score)
        all_histories = redis_cache.get_medical_history(current_user['user_id'])
        if isinstance(all_histories, list):
            # Return only the latest created history for frontend compatibility
            latest_history = all_histories[-1] if all_histories else None
            return {"histories": [latest_history] if latest_history else []}
        return {"histories": all_histories}
    except HTTPException as he:
        logger.error(f"HTTP error creating medical history: {he.detail}")
        raise
    except Exception as e:
        logger.error(f"Error creating medical history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/me/history", response_model=MedicalHistoryListResponse)
async def get_current_medical_history(current_user: Dict = Depends(get_current_user)):
    try:
        if current_user['role'] != 'patient':
            raise HTTPException(status_code=403, detail="Only patients can view their own data")

        redis_cache = RedisCache()
        histories = redis_cache.get_medical_history(current_user['user_id'])
        if not histories:
            raise HTTPException(status_code=404, detail="Medical history not found")

        redis_cache.log_audit(
            action="view_medical_history",
            user_id=current_user['user_id'],
            details={"user_id": current_user['user_id']}
        )

        logger.info(f"Retrieved medical histories for patient: {current_user['user_id']}")
        return {"histories": histories}
    except HTTPException as he:
        logger.error(f"HTTP error getting medical history: {he.detail}")
        raise
    except Exception as e:
        logger.error(f"Error getting medical history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/me/history", response_model=Dict)
async def update_current_medical_history(image: ImageUpdatePatient, 
                                        current_user: Dict = Depends(get_current_user)):
    try:
        if current_user['role'] != 'patient':
            raise HTTPException(status_code=403, detail="Only patients can manage their own data")

        image_data = image.dict(exclude_unset=True)
        image_data['comment'] = None
        image_data['diagnosis_score'] = None

        medical_schema = MedicalHistorySchema()
        try:
            medical_schema.load(image_data, partial=True)
        except ValidationError as ve:
            raise HTTPException(status_code=400, detail=f"Invalid medical history data: {ve.messages}. Please correct the data and try again.")

        redis_cache = RedisCache()
        existing_history = redis_cache.get_medical_history(current_user['user_id'])
        if not existing_history:
            raise HTTPException(status_code=404, detail="Medical history not found")

        image_data['user_id'] = current_user['user_id']
        image_data['image_id'] = existing_history['image_id']
        existing_dates = existing_history.get('date', [])
        if existing_dates:
            existing_dates = [
                d.isoformat() if isinstance(d, datetime) else d
                for d in existing_dates
            ]
        else:
            existing_dates = []
        image_data['date'] = existing_dates + [datetime.now(timezone.utc).isoformat()]

        if not redis_cache.update_medical_history(current_user['user_id'], image_data):
            raise HTTPException(status_code=400, detail="Failed to update medical history")

        updated_history = redis_cache.get_medical_history(current_user['user_id'])
        if 'date' in updated_history and updated_history['date']:
            updated_history['date'] = [
                d.isoformat() if isinstance(d, datetime) else d
                for d in updated_history['date']
            ]
        else:
            updated_history['date'] = []

        redis_cache.log_audit(
            action="update_medical_history",
            user_id=current_user['user_id'],
            details={"user_id": current_user['user_id'], "image_id": image_data['image_id']}
        )

        logger.info(f"Updated medical history for patient {current_user['user_id']}")
        return updated_history
    except HTTPException as he:
        logger.error(f"HTTP error updating medical history: {he.detail}")
        raise
    except Exception as e:
        logger.error(f"Error updating medical history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/get/{user_id}", response_model=Dict)
async def get_patient(user_id: str, current_user: Dict = Depends(get_current_user)):
    try:
        if current_user['role'] == 'patient' and current_user['user_id'] != user_id:
            raise HTTPException(status_code=403, detail="Patients can only view their own data")

        redis_cache = RedisCache()
        patient = redis_cache.get_patient_by_id(user_id)
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")

        redis_cache.log_audit(
            action="view_patient",
            user_id=current_user['user_id'],
            details={"viewed_user_id": user_id}
        )

        logger.info(f"Retrieved patient: {user_id}")
        return patient
    except HTTPException as he:
        logger.error(f"HTTP error getting patient: {he.detail}")
        raise
    except Exception as e:
        logger.error(f"Error getting patient: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/update/{user_id}", response_model=Dict)
async def update_patient(user_id: str, patient: PatientUpdate, 
                        current_user: Dict = Depends(get_current_user)):
    try:
        if current_user['role'] == 'patient' and current_user['user_id'] != user_id:
            raise HTTPException(status_code=403, detail="Patients can only update their own data")

        patient_data = patient.dict(exclude_unset=True)
        patient_schema = PatientSchema()
        try:
            patient_schema.load(patient_data, partial=True)
        except ValidationError as ve:
            raise HTTPException(status_code=400, detail=f"Invalid input: {ve.messages}. Please correct the data and try again.")

        if 'birthdate' in patient_data and patient_data['birthdate']:
            is_valid, error_message = validate_age(patient_data['birthdate'])
            if not is_valid:
                raise HTTPException(status_code=400, detail=f"Invalid birthdate: {error_message}. Please correct and try again.")

        redis_cache = RedisCache()
        if not redis_cache.update_patient(user_id, patient_data):
            raise HTTPException(status_code=404, detail="Patient not found")

        updated_patient = redis_cache.get_patient_by_id(user_id)
        redis_cache.log_audit(
            action="update_patient",
            user_id=current_user['user_id'],
            details={"user_id": user_id}
        )

        logger.info(f"Updated patient: {user_id}")
        return updated_patient
    except HTTPException as he:
        logger.error(f"HTTP error updating patient: {he.detail}")
        raise
    except Exception as e:
        logger.error(f"Error updating patient: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{user_id}/history", response_model=MedicalHistoryListResponse)
async def create_medical_history(user_id: str, image: ImageCreatePatient, 
                                current_user: Dict = Depends(get_current_user)):
    try:
        if current_user['role'] == 'patient' and current_user['user_id'] != user_id:
            raise HTTPException(status_code=403, detail="Patients can only manage their own data")

        image_data = image.dict()
        image_data['user_id'] = user_id
        image_data['comment'] = None
        image_data['diagnosis_score'] = None

        medical_schema = MedicalHistorySchema()
        try:
            validated_data = medical_schema.load(image_data)
        except ValidationError as ve:
            raise HTTPException(status_code=400, detail=f"Invalid medical history data: {ve.messages}. Please correct the data and try again.")

        redis_cache = RedisCache()
        validated_data['date'] = [datetime.now(timezone.utc).isoformat()]
        if not redis_cache.create_medical_history(validated_data):
            raise HTTPException(status_code=400, detail="Failed to add medical history")

        redis_cache.log_audit(
            action="create_medical_history",
            user_id=current_user['user_id'],
            details={"user_id": user_id, "image_id": validated_data['image_id']}
        )

        logger.info(f"Created medical history {validated_data['image_id']} for patient {user_id}")
        # Return the updated list of all histories for the user
        all_histories = redis_cache.get_medical_history(user_id)
        return {"histories": all_histories}
    except HTTPException as he:
        logger.error(f"HTTP error creating medical history: {he.detail}")
        raise
    except Exception as e:
        logger.error(f"Error creating medical history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{user_id}/history", response_model=MedicalHistoryListResponse)
async def get_medical_history(user_id: str, current_user: Dict = Depends(get_current_user)):
    try:
        if current_user['role'] == 'patient' and current_user['user_id'] != user_id:
            raise HTTPException(status_code=403, detail="Patients can only view their own data")

        redis_cache = RedisCache()
        histories = redis_cache.get_medical_history(user_id)
        if not histories:
            raise HTTPException(status_code=404, detail="Medical history not found")

        redis_cache.log_audit(
            action="view_medical_history",
            user_id=current_user['user_id'],
            details={"user_id": user_id}
        )

        logger.info(f"Retrieved medical histories for patient: {user_id}")
        return {"histories": histories}
    except HTTPException as he:
        logger.error(f"HTTP error getting medical history: {he.detail}")
        raise
    except Exception as e:
        logger.error(f"Error getting medical history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{user_id}/history", response_model=Dict)
async def update_medical_history(user_id: str, image: ImageUpdatePatient, 
                                current_user: Dict = Depends(get_current_user)):
    try:
        if current_user['role'] == 'patient' and current_user['user_id'] != user_id:
            raise HTTPException(status_code=403, detail="Patients can only manage their own data")

        image_data = image.dict(exclude_unset=True)
        image_data['comment'] = None
        image_data['diagnosis_score'] = None

        medical_schema = MedicalHistorySchema()
        try:
            medical_schema.load(image_data, partial=True)
        except ValidationError as ve:
            raise HTTPException(status_code=400, detail=f"Invalid medical history data: {ve.messages}. Please correct the data and try again.")

        redis_cache = RedisCache()
        existing_history = redis_cache.get_medical_history(user_id)
        if not existing_history:
            raise HTTPException(status_code=404, detail="Medical history not found")

        image_data['user_id'] = user_id
        image_data['image_id'] = existing_history['image_id']
        existing_dates = existing_history.get('date', [])
        if existing_dates:
            existing_dates = [
                d.isoformat() if isinstance(d, datetime) else d
                for d in existing_dates
            ]
        else:
            existing_dates = []
        image_data['date'] = existing_dates + [datetime.now(timezone.utc).isoformat()]

        if not redis_cache.update_medical_history(user_id, image_data):
            raise HTTPException(status_code=400, detail="Failed to update medical history")

        updated_history = redis_cache.get_medical_history(user_id)
        if 'date' in updated_history and updated_history['date']:
            updated_history['date'] = [
                d.isoformat() if isinstance(d, datetime) else d
                for d in updated_history['date']
            ]
        else:
            updated_history['date'] = []

        redis_cache.log_audit(
            action="update_medical_history",
            user_id=current_user['user_id'],
            details={"user_id": user_id, "image_id": image_data['image_id']}
        )

        logger.info(f"Updated medical history for patient {user_id}")
        return updated_history
    except HTTPException as he:
        logger.error(f"HTTP error updating medical history: {he.detail}")
        raise
    except Exception as e:
        logger.error(f"Error updating medical history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))