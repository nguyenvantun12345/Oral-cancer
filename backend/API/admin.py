from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime, timedelta, timezone
import logging
import asyncio
from auth_utils import hash_password
from db_redis import RedisCache
from api import get_current_user, PatientCreate, PatientUpdate
from marshmallow import ValidationError
from db_schema import PatientSchema, MedicalHistorySchema

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["Admin"])

class BulkPatientCreate(BaseModel):
    patients: List[PatientCreate]

class BulkPatientUpdate(BaseModel):
    updates: List[Dict[str, str]]  # user_id or username or name and update data
    medical_histories: Optional[List[Dict]]  # Optional medical history updates

class DeleteRequest(BaseModel):
    user_ids: List[str]
    wait_seconds: int = 30
    max_delete: int = 15

@router.post("/patients/bulk-create", response_model=Dict)
async def create_multiple_patients(bulk_data: BulkPatientCreate, current_user: Dict = Depends(get_current_user)):
    try:
        if current_user['role'] != 'admin':
            raise HTTPException(status_code=403, detail="Only admins can create patients")

        patient_schema = PatientSchema()
        redis_cache = RedisCache()
        validated_patients = []
        
        for patient in bulk_data.patients:
            patient_data = patient.dict()
            patient_data['role'] = 'patient'
            try:
                validated_data = patient_schema.load(patient_data, partial=True)
                validated_data['password'] = hash_password(validated_data['password'])
                validated_patients.append(validated_data)
            except ValidationError as ve:
                raise HTTPException(status_code=400, detail=f"Invalid patient data: {ve.messages}")

        user_ids = redis_cache.create_patient(validated_patients)
        if not user_ids:
            raise HTTPException(status_code=400, detail="Failed to create patients")

        redis_cache.log_audit(
            action="bulk_create_patients",
            user_id=current_user['user_id'],
            details={"created_count": len(user_ids)}
        )

        logger.info(f"Created {len(user_ids)} patients by admin {current_user['username']}")
        return {"message": f"Created {len(user_ids)} patients", "user_ids": user_ids}
    except HTTPException as he:
        logger.error(f"HTTP error creating patients: {he.detail}")
        raise
    except Exception as e:
        logger.error(f"Error creating patients: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/patients/bulk-update", response_model=Dict)
async def update_multiple_patients(bulk_data: BulkPatientUpdate, current_user: Dict = Depends(get_current_user)):
    try:
        if current_user['role'] != 'admin':
            raise HTTPException(status_code=403, detail="Only admins can update patients")

        redis_cache = RedisCache()
        patient_schema = PatientSchema()
        medical_schema = MedicalHistorySchema()
        updated_patients = []
        updated_histories = []

        # Update patients
        for update in bulk_data.updates:
            identifier = update.get('user_id') or update.get('username') or update.get('name')
            if not identifier:
                logger.warning("No valid identifier provided for patient update")
                continue

            patient_data = {k: v for k, v in update.items() if k not in ['user_id', 'username', 'name']}
            
            # Find user_id based on identifier
            user_id = None
            if update.get('user_id'):
                user_id = update['user_id']
            elif update.get('username'):
                user = redis_cache.get_cached_user(update['username'])
                user_id = user['user_id'] if user else None
            elif update.get('name'):
                patients = redis_cache.search_patients(update['name'], limit=1)
                user_id = patients[0]['user_id'] if patients else None

            if not user_id:
                logger.warning(f"No patient found for identifier: {identifier}")
                continue

            try:
                patient_schema.load(patient_data, partial=True)
                if redis_cache.update_patient(user_id, patient_data):
                    updated_patients.append(user_id)
            except ValidationError as ve:
                logger.warning(f"Invalid update data for patient {user_id}: {ve.messages}")
                continue

        # Update medical histories if provided
        if bulk_data.medical_histories:
            for history in bulk_data.medical_histories:
                user_id = history.get('user_id')
                history_data = {k: v for k, v in history.items() if k != 'user_id'}
                history_data['date'] = [datetime.now(timezone.utc).isoformat()]
                
                try:
                    medical_schema.load(history_data, partial=True)
                    if redis_cache.update_medical_history(user_id, history_data):
                        updated_histories.append(user_id)
                except ValidationError as ve:
                    logger.warning(f"Invalid medical history data for patient {user_id}: {ve.messages}")
                    continue

        redis_cache.log_audit(
            action="bulk_update_patients",
            user_id=current_user['user_id'],
            details={
                "updated_patients": updated_patients,
                "updated_histories": updated_histories
            }
        )

        logger.info(f"Updated {len(updated_patients)} patients and {len(updated_histories)} medical histories")
        return {
            "message": "Bulk update completed",
            "updated_patients": updated_patients,
            "updated_histories": updated_histories
        }
    except HTTPException as he:
        logger.error(f"HTTP error updating patients: {he.detail}")
        raise
    except Exception as e:
        logger.error(f"Error updating patients: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/patients/bulk-read", response_model=List[Dict])
async def read_multiple_patients(search_text: str = "", limit: int = 10, skip: int = 0, 
                               current_user: Dict = Depends(get_current_user)):
    try:
        if current_user['role'] != 'admin':
            raise HTTPException(status_code=403, detail="Only admins can read patients")

        redis_cache = RedisCache()
        patients = redis_cache.search_patients(search_text, limit, skip)
        
        # Fetch medical history for each patient
        for patient in patients:
            history = redis_cache.get_medical_history(patient['user_id'])
            patient['medical_history'] = history if history else {}

        redis_cache.log_audit(
            action="bulk_read_patients",
            user_id=current_user['user_id'],
            details={"search_text": search_text, "limit": limit, "skip": skip}
        )

        logger.info(f"Retrieved {len(patients)} patients with medical histories")
        return patients
    except HTTPException as he:
        logger.error(f"HTTP error reading patients: {he.detail}")
        raise
    except Exception as e:
        logger.error(f"Error reading patients: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

from starlette.concurrency import run_in_threadpool

@router.delete("/patients/bulk-delete", response_model=Dict)
async def delete_multiple_patients(
    delete_request: DeleteRequest,
    current_user: Dict = Depends(get_current_user)
):
    try:
        if current_user['role'] != 'admin':
            raise HTTPException(status_code=403, detail="Only admins can delete patients")

        if len(delete_request.user_ids) > delete_request.max_delete:
            raise HTTPException(status_code=400, detail=f"Cannot delete more than {delete_request.max_delete} patients at once")

        redis_cache = RedisCache()
        deleted_ids = []

        logger.info(f"Starting deletion process with {delete_request.wait_seconds}s wait period")
        await asyncio.sleep(delete_request.wait_seconds)  # Wait period for cancellation

        for user_id in delete_request.user_ids:
            deleted = await run_in_threadpool(redis_cache.delete_patient, user_id)
            if deleted:
                deleted_ids.append(user_id)
                await run_in_threadpool(redis_cache.delete_medical_history, user_id)

        await run_in_threadpool(
            redis_cache.log_audit,
            action="bulk_delete_patients",
            user_id=current_user['user_id'],
            details={"deleted_ids": deleted_ids, "wait_seconds": delete_request.wait_seconds}
        )

        logger.info(f"Deleted {len(deleted_ids)} patients")
        return {"message": f"Deleted {len(deleted_ids)} patients", "deleted_ids": deleted_ids}

    except HTTPException as he:
        logger.error(f"HTTP error deleting patients: {he.detail}")
        raise
    except Exception as e:
        logger.error(f"Error deleting patients: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
