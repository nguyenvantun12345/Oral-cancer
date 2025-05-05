from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, Optional
from datetime import datetime, timedelta, timezone
import jwt
import logging
import os
from auth_utils import hash_password, verify_password
from checkquailty import process_multiple_images, load_image
from solvequality import enhance_image
from data_utils import fill_missing_data, aggregate_and_visualize
from marshmallow import ValidationError
from db_redis import RedisCache
from pydantic import ValidationError as PydanticValidationError
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    error_messages = [err['msg'] for err in exc.errors() if err['msg']]
    if any("birthdate must be in format dd/mm/yyyy" in msg for msg in error_messages):
        error_messages = ["birthdate must be in format dd/mm/yyyy"]
    return JSONResponse(
        status_code=400,
        content={"detail": error_messages}
    )

SECRET_KEY = os.getenv("JWT_SECRET", "f7b3e8c2a9d4f1e6b0c7a8d5e2f3b9c0a1d4e7f8")
if not os.getenv("JWT_SECRET"):
    logger.warning("JWT_SECRET not set in environment variables, using default key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

REFERENCE_EXCEL = os.path.join(os.getenv("OUTPUT_DIR", "/tmp/patient_data"), "report_latest.xlsx")

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

def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")
        redis_cache = RedisCache()
        user = redis_cache.get_cached_user(username)
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/patients", response_model=Dict)
async def create_patient(patient: PatientCreate):
    try:
        patient_data = patient.dict(exclude_unset=True)
        patient_data = fill_missing_data(patient_data, REFERENCE_EXCEL)
        patient_data['password'] = hash_password(patient_data['password'])
        patient_data['user_id'] = f"USR{int(datetime.now(timezone.utc).timestamp())}"
        redis_cache = RedisCache()
        user_ids = redis_cache.create_patient([patient_data])
        if not user_ids:
            raise HTTPException(status_code=400, detail="Failed to create patient")
        patients = redis_cache.search_patients("", limit=100)
        report = aggregate_and_visualize(patients, f"patient_create_{patient_data['user_id']}")
        logger.info(f"Created patient: {patient_data['username']}")
        return {**patient_data, "outlier_report": report}
    except (ValidationError, PydanticValidationError) as ve:
        error_messages = (
            ve.messages if isinstance(ve, ValidationError)
            else [err['msg'] for err in ve.errors() if err['msg']] or ["Invalid input"]
        )
        if any("birthdate must be in format dd/mm/yyyy" in msg for msg in error_messages):
            error_messages = ["birthdate must be in format dd/mm/yyyy"]
        raise HTTPException(status_code=400, detail=error_messages)
    except Exception as e:
        logger.error(f"Error creating patient: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/patients/{user_id}", response_model=Dict)
async def get_patient(user_id: str, current_user: Dict = Depends(get_current_user)):
    try:
        redis_cache = RedisCache()
        patient = redis_cache.get_patient_by_id(user_id)
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        if current_user['role'] != 'admin' and current_user['user_id'] != user_id:
            raise HTTPException(status_code=403, detail="Not authorized")
        logger.info(f"Retrieved patient: {user_id}")
        return patient
    except Exception as e:
        logger.error(f"Error getting patient: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/patients/{user_id}", response_model=Dict)
async def update_patient(user_id: str, patient: PatientUpdate, 
                        current_user: Dict = Depends(get_current_user)):
    try:
        if current_user['role'] != 'admin' and current_user['user_id'] != user_id:
            raise HTTPException(status_code=403, detail="Not authorized")
        patient_data = patient.dict(exclude_unset=True)
        patient_data = fill_missing_data(patient_data, REFERENCE_EXCEL)
        redis_cache = RedisCache()
        if not redis_cache.update_patient(user_id, patient_data):
            raise HTTPException(status_code=404, detail="Patient not found")
        patients = redis_cache.search_patients("", limit=100)
        report = aggregate_and_visualize(patients, f"patient_update_{user_id}")
        logger.info(f"Updated patient: {user_id}")
        return {**patient_data, "outlier_report": report}
    except ValidationError as ve:
        raise HTTPException(status_code=400, detail=ve.messages)
    except Exception as e:
        logger.error(f"Error updating patient: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/patients/{user_id}", response_model=Dict)
async def delete_patient(user_id: str, current_user: Dict = Depends(get_current_user)):
    try:
        if current_user['role'] != 'admin':
            raise HTTPException(status_code=403, detail="Not authorized")
        redis_cache = RedisCache()
        if not redis_cache.delete_patient(user_id):
            raise HTTPException(status_code=404, detail="Patient not found")
        logger.info(f"Deleted patient: {user_id}")
        return {"message": "Patient deleted"}
    except Exception as e:
        logger.error(f"Error deleting patient: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/patients/{user_id}/history", response_model=Dict)
async def create_medical_history(user_id: str, image: ImageCreate, 
                                current_user: Dict = Depends(get_current_user)):
    try:
        if current_user['role'] != 'admin' and current_user['user_id'] != user_id:
            raise HTTPException(status_code=403, detail="Not authorized")
        image_data = image.dict()
        image_data['user_id'] = user_id
        
        # Validate and load image
        logger.info(f"Attempting to load image: {image_data['image']}")
        img = load_image(image_data['image'])
        if img is None:
            logger.error(f"Image loading failed for {image_data['image']}")
            raise HTTPException(status_code=400, detail=f"Failed to load image from {image_data['image']}")

        logger.info(f"Processing image: {image_data['image']}")
        quality_result = process_multiple_images([image_data['image']])
        if not quality_result or not quality_result.values():
            logger.error(f"Image quality evaluation failed for {image_data['image']}")
            raise HTTPException(status_code=400, detail="Failed to evaluate image quality")
        quality_result = list(quality_result.values())[0]
        
        if quality_result.loc['Deviation'].sum() > 0:
            logger.info(f"Image {image_data['image_id']} does not meet quality standards, enhancing...")
            enhanced_image = enhance_image(image_data['image'])
            if not enhanced_image or 'output_path' not in enhanced_image:
                logger.error(f"Image enhancement failed for {image_data['image']}")
                raise HTTPException(status_code=400, detail="Image enhancement failed")
            image_data['image'] = enhanced_image['output_path']
            logger.info(f"Re-processing enhanced image: {image_data['image']}")
            quality_result = process_multiple_images([image_data['image']])
            if not quality_result or not quality_result.values():
                logger.error(f"Enhanced image quality evaluation failed for {image_data['image']}")
                raise HTTPException(status_code=400, detail="Failed to evaluate enhanced image quality")
            quality_result = list(quality_result.values())[0]
            if quality_result.loc['Deviation'].sum() > 0:
                logger.error(f"Enhanced image quality still below standards for {image_data['image']}")
                raise HTTPException(status_code=400, detail="Image quality still below standards after enhancement")

        redis_cache = RedisCache()
        image_data['quality_report'] = quality_result.to_dict()
        # Add the date field manually as a list of ISO strings to avoid serialization issues
        image_data['date'] = [datetime.now(timezone.utc).isoformat()]
        if not redis_cache.create_medical_history(image_data):
            logger.error(f"Failed to add medical history for user_id {user_id}")
            raise HTTPException(status_code=400, detail="Failed to add medical history")
        logger.info(f"Created medical history {image_data['image_id']} for patient {user_id}")
        return image_data
    except HTTPException as he:
        logger.error(f"HTTP error creating medical history: {str(he.detail)}")
        raise
    except Exception as e:
        logger.error(f"Error creating medical history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/patients/{user_id}/history", response_model=Dict)
async def get_medical_history(user_id: str, current_user: Dict = Depends(get_current_user)):
    try:
        if current_user['role'] != 'admin' and current_user['user_id'] != user_id:
            raise HTTPException(status_code=403, detail="Not authorized")
        redis_cache = RedisCache()
        history = redis_cache.get_medical_history(user_id)
        if not history:
            raise HTTPException(status_code=404, detail="Medical history not found")
        logger.info(f"Retrieved medical history for patient: {user_id}")
        return history
    except Exception as e:
        logger.error(f"Error getting medical history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/patients/{user_id}/history", response_model=Dict)
async def update_medical_history(user_id: str, image: ImageUpdate, 
                                current_user: Dict = Depends(get_current_user)):
    try:
        if current_user['role'] != 'admin' and current_user['user_id'] != user_id:
            raise HTTPException(status_code=403, detail="Not authorized")
        image_data = image.dict(exclude_unset=True)
        if 'image' in image_data:
            logger.info(f"Attempting to load image: {image_data['image']}")
            img = load_image(image_data['image'])
            if img is None:
                logger.error(f"Image loading failed for {image_data['image']}")
                raise HTTPException(status_code=400, detail=f"Failed to load image from {image_data['image']}")
            logger.info(f"Processing image: {image_data['image']}")
            quality_result = process_multiple_images([image_data['image']])
            if not quality_result or not quality_result.values():
                logger.error(f"Image quality evaluation failed for {image_data['image']}")
                raise HTTPException(status_code=400, detail="Failed to evaluate image quality")
            quality_result = list(quality_result.values())[0]
            if quality_result.loc['Deviation'].sum() > 0:
                logger.info(f"Image for user_id {user_id} does not meet quality standards, enhancing...")
                enhanced_image = enhance_image(image_data['image'])
                if not enhanced_image or 'output_path' not in enhanced_image:
                    logger.error(f"Image enhancement failed for {image_data['image']}")
                    raise HTTPException(status_code=400, detail="Image enhancement failed")
                image_data['image'] = enhanced_image['output_path']
                logger.info(f"Re-processing enhanced image: {image_data['image']}")
                quality_result = process_multiple_images([image_data['image']])
                if not quality_result or not quality_result.values():
                    logger.error(f"Enhanced image quality evaluation failed for {image_data['image']}")
                    raise HTTPException(status_code=400, detail="Failed to evaluate enhanced image quality")
                quality_result = list(quality_result.values())[0]
                if quality_result.loc['Deviation'].sum() > 0:
                    logger.error(f"Enhanced image quality still below standards for {image_data['image']}")
                    raise HTTPException(status_code=400, detail="Image quality still below standards after enhancement")
                image_data['quality_report'] = quality_result.to_dict()
        redis_cache = RedisCache()
        existing_history = redis_cache.get_medical_history(user_id)
        if not existing_history:
            raise HTTPException(status_code=404, detail="Medical history not found")
        image_data['user_id'] = user_id
        image_data['image_id'] = existing_history['image_id']
        # Convert existing dates to ISO strings
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
            logger.error(f"Failed to update medical history for user_id {user_id}")
            raise HTTPException(status_code=400, detail="Failed to update medical history")
        # Fetch the updated history and ensure datetime is converted to ISO string
        updated_history = redis_cache.get_medical_history(user_id)
        if 'date' in updated_history and updated_history['date']:
            updated_history['date'] = [
                d.isoformat() if isinstance(d, datetime) else d
                for d in updated_history['date']
            ]
        else:
            updated_history['date'] = []
        logger.info(f"Updated medical history for patient {user_id}")
        return updated_history
    except HTTPException as he:
        logger.error(f"HTTP error updating medical history: {str(he.detail)}")
        raise
    except Exception as e:
        logger.error(f"Error updating medical history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/patients/{user_id}/history", response_model=Dict)
async def delete_medical_history(user_id: str, current_user: Dict = Depends(get_current_user)):
    try:
        if current_user['role'] != 'admin':
            raise HTTPException(status_code=403, detail="Not authorized")
        redis_cache = RedisCache()
        if not redis_cache.delete_medical_history(user_id):
            raise HTTPException(status_code=404, detail="Medical history not found")
        logger.info(f"Deleted medical history for patient {user_id}")
        return {"message": "Medical history deleted"}
    except Exception as e:
        logger.error(f"Error deleting medical history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/register", response_model=Dict)
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
        raise HTTPException(status_code=400, detail=ve.messages)
    except Exception as e:
        logger.error(f"Error registering user: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/login", response_model=Dict)
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

if __name__ == "__main__":
    from fastapi.testclient import TestClient
    import uuid
    from redis.exceptions import ConnectionError as RedisConnectionError
    from unittest.mock import patch  # Add for mocking

    client = TestClient(app)
    unique_id = str(uuid.uuid4())[:8]
    sample_patient = {
        'username': f'testuser_{unique_id}',
        'password': 'testpassword',
        'name': 'Test User',
        'birthdate': '01/01/1990',
        'gender': 'male',
        'role': 'patient',
        'work': 'Engineer',
        'email': f'test_{unique_id}@example.com'
    }
    sample_medical_history = {
        'image_id': f'img_{unique_id}',
        'image': 'https://d1hjkbq40fs2x4.cloudfront.net/2016-01-31/files/1045.jpg',
        'diagnosis_score': 0.85,
        'comment': 'Test medical history'
    }

    redis_cache = None
    try:
        redis_cache = RedisCache(ttl_seconds=3600)
    except RedisConnectionError as e:
        print(f"Không thể kết nối Redis: {str(e)}. Thoát test.")
        exit(1)
    except Exception as e:
        print(f"Lỗi khởi tạo RedisCache: {str(e)}. Thoát test.")
        exit(1)

    def cleanup(user_id):
        try:
            redis_cache.delete_patient(user_id)
            redis_cache.delete_medical_history(user_id)
            redis_cache.invalidate_cache(f"user:*{user_id}*")
            redis_cache.invalidate_cache(f"patient:{user_id}")
            redis_cache.invalidate_cache(f"medical:{user_id}:*")
            print(f"Đã dọn dẹp dữ liệu cho user_id {user_id}")
        except Exception as e:
            print(f"Lỗi khi dọn dẹp: {str(e)}")

    print("Test 1: Đăng ký tài khoản mới")
    try:
        response = client.post("/register", json=sample_patient)
        assert response.status_code == 200
        assert response.json()['username'] == sample_patient['username']
        assert response.json()['message'] == "User registered"
        print("Test 1 passed")
    except Exception as e:
        print(f"Test 1 failed: {str(e)}")
        cleanup(sample_patient.get('user_id', f"USR{int(datetime.now(timezone.utc).timestamp())}"))
        exit(1)

    print("Test 2: Đăng ký với email không hợp lệ")
    try:
        invalid_patient = sample_patient.copy()
        invalid_patient['email'] = 'invalid_email'
        invalid_patient['username'] = f'invaliduser_{unique_id}'
        response = client.post("/register", json=invalid_patient)
        assert response.status_code == 400
        print("Test 2 passed")
    except Exception as e:
        print(f"Test 2 failed: {str(e)}")
        cleanup(f"USR{int(datetime.now(timezone.utc).timestamp())}")
        exit(1)

    print("Test 3: Đăng nhập thành công")
    try:
        login_data = {
            'username': sample_patient['username'],
            'password': sample_patient['password']
        }
        response = client.post("/login", data=login_data)
        assert response.status_code == 200
        assert 'access_token' in response.json()
        assert response.json()['token_type'] == 'bearer'
        token = response.json()['access_token']
        print("Test 3 passed")
    except Exception as e:
        print(f"Test 3 failed: {str(e)}")
        cleanup(sample_patient.get('user_id', f"USR{int(datetime.now(timezone.utc).timestamp())}"))
        exit(1)

    print("Test 4: Đăng nhập với thông tin sai")
    try:
        login_data = {
            'username': sample_patient['username'],
            'password': 'wrongpassword'
        }
        response = client.post("/login", data=login_data)
        assert response.status_code == 401
        assert response.json()['detail'] == "Invalid credentials"
        print("Test 4 passed")
    except Exception as e:
        print(f"Test 4 failed: {str(e)}")
        cleanup(sample_patient.get('user_id', f"USR{int(datetime.now(timezone.utc).timestamp())}"))
        exit(1)

    print("Test 5: Tạo bệnh nhân mới")
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = client.post("/patients", json=sample_patient, headers=headers)
        assert response.status_code == 200
        assert response.json()['username'] == sample_patient['username']
        assert 'outlier_report' in response.json()
        user_id = response.json()['user_id']
        print("Test 5 passed")
    except Exception as e:
        print(f"Test 5 failed: {str(e)}")
        cleanup(sample_patient.get('user_id', f"USR{int(datetime.now(timezone.utc).timestamp())}"))
        exit(1)

    print("Test 6: Tạo bệnh nhân với birthdate không hợp lệ")
    try:
        invalid_patient = sample_patient.copy()
        invalid_patient['birthdate'] = '2025-01-01'
        headers = {"Authorization": f"Bearer {token}"}
        response = client.post("/patients", json=invalid_patient, headers=headers)
        print(f"Response: {response.status_code} {response.json()}")
        assert response.status_code == 400
        print("Test 6 passed")
    except Exception as e:
        print(f"Test 6 failed: {str(e)}")
        cleanup(user_id)
        exit(1)

    print("Test 7: Lấy thông tin bệnh nhân")
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = client.get(f"/patients/{user_id}", headers=headers)
        assert response.status_code == 200
        assert response.json()['username'] == sample_patient['username']
        print("Test 7 passed")
    except Exception as e:
        print(f"Test 7 failed: {str(e)}")
        cleanup(user_id)
        exit(1)

    print("Test 8: Cập nhật thông tin bệnh nhân")
    try:
        update_data = {'name': 'Updated User'}
        headers = {"Authorization": f"Bearer {token}"}
        response = client.put(f"/patients/{user_id}", json=update_data, headers=headers)
        assert response.status_code == 200
        assert response.json()['name'] == 'Updated User'
        print("Test 8 passed")
    except Exception as e:
        print(f"Test 8 failed: {str(e)}")
        cleanup(user_id)
        exit(1)

    print("Test 9: Tạo lịch sử y tế")
    try:
        headers = {"Authorization": f"Bearer {token}"}
        # Mock process_multiple_images to return a valid quality_result
        mock_quality_result = pd.DataFrame({
            'Metric': ['Brightness', 'Contrast', 'Noise'],
            'Value': [100, 50, 10],
            'Deviation': [0, 0, 0]
        }).set_index('Metric')
        with patch('checkquailty.process_multiple_images', return_value={'image1': mock_quality_result}):
            response = client.post(f"/patients/{user_id}/history", json=sample_medical_history, headers=headers)
        assert response.status_code == 200
        assert response.json()['image_id'] == sample_medical_history['image_id']
        print("Test 9 passed")
    except Exception as e:
        print(f"Test 9 failed: {str(e)}")
        cleanup(user_id)
        exit(1)

    print("Test 10: Lấy lịch sử y tế")
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = client.get(f"/patients/{user_id}/history", headers=headers)
        assert response.status_code == 200
        assert response.json()['image_id'] == sample_medical_history['image_id']
        print("Test 10 passed")
    except Exception as e:
        print(f"Test 10 failed: {str(e)}")
        cleanup(user_id)
        exit(1)

    print("Test 11: Cập nhật lịch sử y tế")
    try:
        # Update sample_medical_history with the correct user_id
        sample_medical_history['user_id'] = user_id
        update_medical = {'diagnosis_score': 0.9}
        headers = {"Authorization": f"Bearer {token}"}
        # Mock process_multiple_images to return a valid quality_result for the POST request
        mock_quality_result = pd.DataFrame({
            'Metric': ['Brightness', 'Contrast', 'Noise'],
            'Value': [100, 50, 10],
            'Deviation': [0, 0, 0]
        }).set_index('Metric')
        with patch('checkquailty.process_multiple_images', return_value={'image1': mock_quality_result}):
            # Delete existing medical history to avoid unique index violation
            redis_cache.delete_medical_history(user_id)
            response = client.post(f"/patients/{user_id}/history", json=sample_medical_history, headers=headers)
            assert response.status_code == 200, f"Failed to create medical history: {response.json()}"
        response = client.put(f"/patients/{user_id}/history", json=update_medical, headers=headers)
        assert response.status_code == 200
        assert response.json()['diagnosis_score'] == 0.9
        print("Test 11 passed")
    except Exception as e:
        print(f"Test 11 failed: {str(e)}")
        cleanup(user_id)
        exit(1)

    print("Test 12: Xóa lịch sử y tế")
    try:
        admin_patient = sample_patient.copy()
        admin_patient['username'] = f'admin_{unique_id}'
        admin_patient['email'] = f'admin_{unique_id}@example.com'
        admin_patient['role'] = 'admin'
        admin_response = client.post("/register", json=admin_patient)
        admin_login = {'username': admin_patient['username'], 'password': admin_patient['password']}
        admin_token_response = client.post("/login", data=admin_login)
        admin_token = admin_token_response.json()['access_token']
        headers = {"Authorization": f"Bearer {admin_token}"}
        response = client.delete(f"/patients/{user_id}/history", headers=headers)
        assert response.status_code == 200
        assert response.json()['message'] == "Medical history deleted"
        print("Test 12 passed")
        cleanup(admin_response.json().get('user_id', f"USR{int(datetime.now(timezone.utc).timestamp())}"))
    except Exception as e:
        print(f"Test 12 failed: {str(e)}")
        cleanup(user_id)
        exit(1)

    print("Test 13: Xóa bệnh nhân")
    try:
        headers = {"Authorization": f"Bearer {admin_token}"}
        response = client.delete(f"/patients/{user_id}", headers=headers)
        assert response.status_code == 200
        assert response.json()['message'] == "Patient deleted"
        print("Test 13 passed")
    except Exception as e:
        print(f"Test 13 failed: {str(e)}")
        cleanup(user_id)
        exit(1)

    cleanup(user_id)
    print("All tests completed!")