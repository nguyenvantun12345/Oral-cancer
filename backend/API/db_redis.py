import redis
import json
import logging
import os
from typing import List, Dict, Optional
from datetime import datetime, timedelta, timezone
from redis.exceptions import ConnectionError as RedisConnectionError
from db_query import Patient, MedicalHistoryCRUD
from db_config import get_database
from bson import ObjectId

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RedisCache:
    def __init__(self, ttl_seconds: int = 7200):
        self.redis = self._connect_redis()
        self.ttl = ttl_seconds
        self.patient_repo = Patient()
        self.history_repo = MedicalHistoryCRUD()

    def _connect_redis(self) -> redis.Redis:
        try:
            redis_client = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                db=int(os.getenv('REDIS_DB', 0)),
                password=os.getenv('REDIS_PASSWORD', None),
                decode_responses=True
            )
            redis_client.ping()
            logger.info("Connected to Redis successfully")
            return redis_client
        except RedisConnectionError as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            raise

    def _serialize_mongo_document(self, data: Dict) -> Dict:
        if not data:
            return data
        serialized = data.copy()
        if '_id' in serialized:
            serialized['_id'] = str(serialized['_id'])
        return serialized

    def create_patient(self, patients: List[Dict]) -> List[str]:
        try:
            user_ids = self.patient_repo.createPatient(patients)
            if not user_ids:
                logger.error("Failed to create patients in MongoDB")
                return []

            for patient in patients:
                user_key = f"user:{patient['username']}"
                patient_key = f"patient:{patient['user_id']}"
                cache_data = {
                    'user_id': patient['user_id'],
                    'username': patient['username'],
                    'password': patient.get('password'),
                    'name': patient.get('name'),
                    'birthdate': patient.get('birthdate'),
                    'gender': patient.get('gender'),
                    'role': patient.get('role'),
                    'work': patient.get('work'),
                    'email': patient.get('email'),
                    'phone': patient.get('phone')
                }
                self.redis.setex(user_key, timedelta(seconds=self.ttl), json.dumps(cache_data))
                self.redis.setex(patient_key, timedelta(seconds=self.ttl), json.dumps(cache_data))
                logger.info(f"Cached patient {patient['username']} in Redis")
            return user_ids
        except Exception as e:
            logger.error(f"Error creating patients: {str(e)}")
            raise

    def get_patient_by_id(self, user_id: str) -> Optional[Dict]:
        patient_key = f"patient:{user_id}"
        cached_data = self.redis.get(patient_key)

        if cached_data:
            logger.info(f"Cache hit for patient {user_id}")
            return json.loads(cached_data)

        logger.info(f"Cache miss for patient {user_id}, querying MongoDB")
        patient = self.patient_repo.get_patient_by_id(user_id)
        if patient:
            cache_data = {
                'user_id': patient['user_id'],
                'username': patient['username'],
                'password': patient.get('password'),
                'name': patient.get('name'),
                'birthdate': patient.get('birthdate'),
                'gender': patient.get('gender'),
                'role': patient.get('role'),
                'work': patient.get('work'),
                'email': patient.get('email'),
                'phone': patient.get('phone')
            }
            self.redis.setex(patient_key, timedelta(seconds=self.ttl), json.dumps(cache_data))
            logger.info(f"Cached patient {user_id} from MongoDB")
            return cache_data
        return None

    def get_cached_user(self, username: str) -> Optional[Dict]:
        user_key = f"user:{username}"
        cached_data = self.redis.get(user_key)
        if cached_data:
            user_data = json.loads(cached_data)
            logger.info(f"Cache hit for user {username}")
            return user_data
        logger.info(f"Cache miss for user {username}, querying MongoDB")
        patients = self.patient_repo.search_patients(username, limit=1)
        if patients:
            patient = patients[0]
            cache_data = {
                'user_id': patient['user_id'],
                'username': patient['username'],
                'password': patient.get('password'),
                'name': patient.get('name'),
                'birthdate': patient.get('birthdate'),
                'gender': patient.get('gender'),
                'role': patient.get('role'),
                'work': patient.get('work'),
                'email': patient.get('email'),
                'phone': patient.get('phone')
            }
            self.redis.setex(user_key, timedelta(seconds=self.ttl), json.dumps(cache_data))
            logger.info(f"Cached user {username} from MongoDB")
            return cache_data
        return None

    def search_patients(self, search_text: str, limit: int = 10, skip: int = 0) -> List[Dict]:
        limit = min(limit, 100)
        cache_key = f"search:patients:{search_text}:{skip}:{limit}"
        cached_data = self.redis.get(cache_key)

        if cached_data:
            logger.info(f"Cache hit for search {search_text}")
            return json.loads(cached_data)

        logger.info(f"Cache miss for search {search_text}, querying MongoDB")
        patients = self.patient_repo.search_patients(search_text, limit, skip)
        if patients:
            cache_data = [
                {
                    'user_id': p['user_id'],
                    'username': p['username'],
                    'name': p.get('name'),
                    'birthdate': p.get('birthdate'),
                    'gender': p.get('gender'),
                    'work': p.get('work'),
                    'email': p.get('email'),
                    'phone': p.get('phone')
                } for p in patients
            ]
            self.redis.setex(cache_key, timedelta(seconds=self.ttl), json.dumps(cache_data))
            logger.info(f"Cached search results for {search_text}")
            return cache_data
        return []

    def update_patient(self, user_id: str, patient_data: Dict) -> bool:
        try:
            success = self.patient_repo.update_patient(user_id, patient_data)
            if not success:
                logger.warning(f"Failed to update patient {user_id} in MongoDB")
                return False

            self.invalidate_cache(f"user:*{user_id}*")
            self.invalidate_cache(f"patient:{user_id}")
            patient = self.patient_repo.get_patient_by_id(user_id)
            if patient:
                cache_data = {
                    'user_id': patient['user_id'],
                    'username': patient['username'],
                    'password': patient.get('password'),
                    'name': patient.get('name'),
                    'birthdate': patient.get('birthdate'),
                    'gender': patient.get('gender'),
                    'role': patient.get('role'),
                    'work': patient.get('work'),
                    'email': patient.get('email'),
                    'phone': patient.get('phone')
                }
                self.redis.setex(f"patient:{user_id}", timedelta(seconds=self.ttl), json.dumps(cache_data))
                self.redis.setex(f"user:{patient['username']}", timedelta(seconds=self.ttl), json.dumps(cache_data))
                logger.info(f"Updated cache for patient {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating patient: {str(e)}")
            return False

    def delete_patient(self, user_id: str) -> bool:
        try:
            success = self.patient_repo.delete_patient(user_id)
            if not success:
                logger.warning(f"Failed to delete patient {user_id} in MongoDB")
                return False

            self.invalidate_cache(f"user:*{user_id}*")
            self.invalidate_cache(f"patient:{user_id}")
            self.invalidate_cache(f"medical:{user_id}:*")
            logger.info(f"Deleted patient {user_id} and cleared cache")
            return True
        except Exception as e:
            logger.error(f"Error deleting patient: {str(e)}")
            return False

    def create_medical_history(self, medical_history: Dict) -> bool:
        try:
            success = self.history_repo.create_medical_history(medical_history)
            if not success:
                logger.error(f"Failed to create medical history for user_id {medical_history['user_id']}")
                return False

            user_id = medical_history['user_id']
            image_key = f"medical:{user_id}:latest"
            serialized_history = self._serialize_mongo_document(medical_history)
            if 'date' in serialized_history:
                serialized_history['date'] = [
                    d.isoformat() if isinstance(d, datetime) else d
                    for d in serialized_history['date']
                ]
            self.redis.setex(image_key, timedelta(seconds=self.ttl), json.dumps(serialized_history))
            logger.info(f"Cached medical history for user_id {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error creating medical history: {str(e)}")
            return False

    def update_medical_history(self, user_id: str, medical_data: Dict) -> bool:
        try:
            success = self.history_repo.update_medical_history(user_id, medical_data)
            if not success:
                logger.warning(f"Failed to update medical history for user_id {user_id}")
                return False

            self.invalidate_cache(f"medical:{user_id}:*")
            medical_data = self.history_repo.get_medical_history_by_user_id(user_id)
            if medical_data:
                serialized_data = self._serialize_mongo_document(medical_data)
                if 'date' in serialized_data:
                    serialized_data['date'] = [
                        d.isoformat() if isinstance(d, datetime) else d
                        for d in serialized_data['date']
                    ]
                image_key = f"medical:{user_id}:latest"
                self.redis.setex(image_key, timedelta(seconds=self.ttl), json.dumps(serialized_data))
                logger.info(f"Updated cache for medical history {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating medical history: {str(e)}")
            return False

    def get_medical_history(self, user_id: str) -> Optional[Dict]:
        image_key = f"medical:{user_id}:latest"
        cached_data = self.redis.get(image_key)

        if cached_data:
            logger.info(f"Cache hit for medical history {user_id}")
            return json.loads(cached_data)

        logger.info(f"Cache miss for medical history {user_id}, querying MongoDB")
        medical_data = self.history_repo.get_medical_history_by_user_id(user_id)
        if medical_data:
            serialized_data = self._serialize_mongo_document(medical_data)
            if 'date' in serialized_data:
                serialized_data['date'] = [
                    d.isoformat() if isinstance(d, datetime) else d
                    for d in serialized_data['date']
                ]
            self.redis.setex(image_key, timedelta(seconds=self.ttl), json.dumps(serialized_data))
            logger.info(f"Cached medical history {user_id} from MongoDB")
            return serialized_data
        return None

    def delete_medical_history(self, user_id: str) -> bool:
        try:
            success = self.history_repo.delete_medical_history(user_id)
            if not success:
                logger.warning(f"Failed to delete medical history for user_id {user_id}")
                return False

            self.invalidate_cache(f"medical:{user_id}:*")
            logger.info(f"Deleted medical history for user_id {user_id} and cleared cache")
            return True
        except Exception as e:
            logger.error(f"Error deleting medical history: {str(e)}")
            return False

    def invalidate_cache(self, pattern: str):
        try:
            cursor = 0
            while True:
                cursor, keys = self.redis.scan(cursor, match=pattern, count=100)
                if keys:
                    self.redis.delete(*keys)
                if cursor == 0:
                    break
            logger.info(f"Invalidated cache for pattern {pattern}")
        except Exception as e:
            logger.error(f"Error invalidating cache: {str(e)}")

    def log_audit(self, action: str, user_id: str = None, details: Dict = None):
        try:
            audit_log = {
                'action': action,
                'user_id': user_id,
                'details': details or {},
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            self.redis.lpush('audit:logs', json.dumps(audit_log))
            self.redis.ltrim('audit:logs', 0, 9999)
            logger.info(f"Audit log created: {action}")
        except Exception as e:
            logger.error(f"Error logging audit: {str(e)}")

    def get_audit_logs(self, limit: int = 100, skip: int = 0) -> List[Dict]:
        try:
            logs = self.redis.lrange('audit:logs', skip, skip + limit - 1)
            return [json.loads(log) for log in logs]
        except Exception as e:
            logger.error(f"Error retrieving audit logs: {str(e)}")
            return []