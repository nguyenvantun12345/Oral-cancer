from fastapi import APIRouter, HTTPException, Depends, Body
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, validator
from typing import Dict, Optional
from datetime import datetime, timedelta, timezone, date
import jwt
import logging
import os
from auth_utils import hash_password, verify_password
from db_redis import RedisCache
from db_schema import PatientSchema
from marshmallow import ValidationError
from redis.exceptions import RedisError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

SECRET_KEY = os.getenv("JWT_SECRET", "f7b3e8c2a9d4f1e6b0c7a8d5e2f3b9c0a1d4e7f8")
if not os.getenv("JWT_SECRET"):
    logger.warning("JWT_SECRET not set in environment variables, using default key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 45

class PatientCreate(BaseModel):
    name: Optional[str]
    birthdate: Optional[str]
    gender: Optional[str]
    work: Optional[str]
    username: str
    email: str
    phone: Optional[str]
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
    work: Optional[str]
    email: Optional[str]
    phone: Optional[str]

    @validator('birthdate', pre=True, always=True)
    def validate_birthdate(cls, value):
        if value:
            try:
                datetime.strptime(value, '%d/%m/%Y')
            except ValueError:
                raise ValueError("birthdate must be in format dd/mm/yyyy")
        return value

class TokenRefresh(BaseModel):
    refresh_token: str

def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if not username or not role:
            raise HTTPException(status_code=401, detail="Invalid token")
        redis_cache = RedisCache()
        user = redis_cache.get_cached_user(username)
        if not user or user['role'] != role:
            raise HTTPException(status_code=401, detail="User not found or role mismatch")
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Access token has expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def validate_age(birthdate: str | datetime | date) -> tuple[bool, str]:
    """Kiểm tra tuổi hợp lệ dựa trên ngày sinh."""
    try:
        if isinstance(birthdate, str):
            parsed_date = datetime.strptime(birthdate.strip(), '%d/%m/%Y')
        elif isinstance(birthdate, (datetime, date)):
            parsed_date = datetime.combine(birthdate, datetime.min.time()) if isinstance(birthdate, date) else birthdate
        else:
            return False, "Invalid birthdate format."

        today = datetime.now()
        age = today.year - parsed_date.year - ((today.month, today.day) < (parsed_date.month, parsed_date.day))
        if age < 0 or age > 120:
            return False, "Age must be between 0 and 120 years."
        return True, ""
    except ValueError:
        return False, "Birthdate must be in format dd/mm/yyyy."

@router.post("/register", response_model=Dict)
async def register(patient: PatientCreate):
    try:
        patient_data = patient.dict()
        patient_data['role'] = 'patient'
        patient_schema = PatientSchema()
        try:
            validated_data = patient_schema.load(patient_data)
        except ValidationError as ve:
            raise HTTPException(status_code=400, detail=f"Invalid input: {ve.messages}. Please correct the data and try again.")

        if 'birthdate' in validated_data and validated_data['birthdate']:
            is_valid, error_message = validate_age(validated_data['birthdate'])
            if not is_valid:
                raise HTTPException(status_code=400, detail=f"Invalid birthdate: {error_message}. Please correct and try again.")

        validated_data['password'] = hash_password(validated_data['password'])
        redis_cache = RedisCache()
        user_ids = redis_cache.create_patient([validated_data])
        if not user_ids:
            raise HTTPException(status_code=400, detail="Failed to register user")

        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = jwt.encode(
            {
                "sub": validated_data['username'],
                "role": validated_data['role'],
                "exp": datetime.now(timezone.utc) + access_token_expires
            },
            SECRET_KEY,
            algorithm=ALGORITHM
        )
        refresh_token = jwt.encode(
            {
                "sub": validated_data['username'],
                "role": validated_data['role'],
                "exp": datetime.now(timezone.utc) + timedelta(days=7)
            },
            SECRET_KEY,
            algorithm=ALGORITHM
        )

        redis_cache.log_audit(
            action="register_patient",
            user_id=user_ids[0],
            details={"username": validated_data['username']}
        )
        
        logger.info(f"Registered and logged in user: {validated_data['username']}")
        return {
            "message": "User registered and logged in",
            "username": validated_data['username'],
            "user_id": user_ids[0],
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
        }
    except ValidationError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid input: {ve.messages}. Please correct the data and try again.")
    except Exception as e:
        logger.error(f"Error registering user: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/login", response_model=Dict)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    try:
        redis_cache = RedisCache()
        user = redis_cache.get_cached_user(form_data.username)
        if not user or not verify_password(form_data.password, user['password']):
            raise HTTPException(status_code=401, detail="Invalid username or password)")

        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = jwt.encode(
            {
                "sub": form_data.username,
                "role": user['role'],
                "exp": datetime.now(timezone.utc) + access_token_expires
            },
            SECRET_KEY,
            algorithm=ALGORITHM
        )
        refresh_token = jwt.encode(
            {
                "sub": form_data.username,
                "role": user['role'],
                "exp": datetime.now(timezone.utc) + timedelta(days=7)
            },
            SECRET_KEY,
            algorithm=ALGORITHM
        )

        redis_cache.log_audit(
            action="login_success",
            user_id=user['user_id'],
            details={"username": form_data.username}
        )

        logger.info(f"User logged in: {form_data.username}")
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "user_id": user['user_id'],
            "username": form_data.username,
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
        }
    except RedisError as re:
        logger.error(f"Redis error during login: {str(re)}")
        raise HTTPException(status_code=500, detail="Cache error")
    except HTTPException as he:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during login: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/refresh-token", response_model=Dict)
async def refresh_token(data: TokenRefresh):
    try:
        payload = jwt.decode(data.refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if not username or not role:
            raise HTTPException(status_code=401, detail="Invalid refresh token")

        redis_cache = RedisCache()
        user = redis_cache.get_cached_user(username)
        if not user or user['role'] != role:
            raise HTTPException(status_code=401, detail="User not found or role mismatch")

        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        new_access_token = jwt.encode(
            {
                "sub": username,
                "role": role,
                "exp": datetime.now(timezone.utc) + access_token_expires
            },
            SECRET_KEY,
            algorithm=ALGORITHM
        )

        redis_cache.log_audit(
            action="refresh_token",
            user_id=user['user_id'],
            details={"username": username}
        )

        logger.info(f"Refreshed token for user: {username}")
        return {
            "access_token": new_access_token,
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
        }
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Refresh token has expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    except Exception as e:
        logger.error(f"Error refreshing token: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/me", response_model=Dict)
async def get_current_user_info(current_user: Dict = Depends(get_current_user)):
    try:
        return {
            "user_id": current_user['user_id'],
            "username": current_user['username'],
            "role": current_user['role'],
            "email": current_user.get('email'),
            "name": current_user.get('name'),
            "birthdate": current_user.get('birthdate'),
            "gender": current_user.get('gender'),
            "phone": current_user.get('phone'),
            "work": current_user.get('work')
        }
    except Exception as e:
        logger.error(f"Error getting current user info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))