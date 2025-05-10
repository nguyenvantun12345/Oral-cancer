# backend/API/auth.py

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import ValidationError
from datetime import datetime, timezone, timedelta
import logging
import jwt
from typing import Dict

from core.config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES, REFERENCE_EXCEL
from core.utils import hash_password, verify_password, fill_missing_data
from core.cache import RedisCache
from core.schemas import PatientCreate  # Or from wherever PatientCreate is defined

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/register", response_model=Dict)
async def register(patient: PatientCreate):
    try:
        patient_data = patient.dict()
        patient_data = fill_missing_data(patient_data, REFERENCE_EXCEL)
        patient_data['password'] = hash_password(patient_data['password'])
        patient_data['user_id'] = f"USR{int(datetime.now(timezone.utc).timestamp())}"
        redis_cache = RedisCache()
        user_ids = redis_cache.create_patient([patient_data])
        if not user_ids:
            raise HTTPException(status_code=400, detail="Failed to register user")
        logger.info(f"Registered user: {patient_data['username']}")
        return {"message": "User registered", "username": patient_data['username']}
    except ValidationError as ve:
        raise HTTPException(status_code=400, detail=ve.errors())
    except Exception as e:
        logger.error(f"Error registering user: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/login", response_model=Dict)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    try:
        redis_cache = RedisCache()
        user = redis_cache.get_cached_user(form_data.username)
        if not user or not verify_password(form_data.password, user['password']):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = jwt.encode(
            {"sub": form_data.username, "exp": datetime.now(timezone.utc) + access_token_expires},
            SECRET_KEY,
            algorithm=ALGORITHM
        )
        logger.info(f"User logged in: {form_data.username}")
        return {"access_token": access_token, "token_type": "bearer"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during login: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
