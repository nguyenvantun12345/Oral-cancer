import redis
import json
import logging
import os
from typing import List, Dict, Optional
from datetime import datetime, timedelta, timezone
from redis.exceptions import ConnectionError as RedisConnectionError
from db_query import Patient, MedicalHistoryCRUD
from db_config import get_database
import pandas as pd
import uuid
from bson import ObjectId

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RedisCache:
    """Lớp quản lý cache Redis làm tầng trung gian giữa API và MongoDB."""

    def __init__(self, ttl_seconds: int = 3600):
        """
        Khởi tạo kết nối Redis.

        Args:
            ttl_seconds: Thời gian sống của cache (giây).
        """
        self.redis = self._connect_redis()
        self.ttl = ttl_seconds
        self.patient_repo = Patient()
        self.history_repo = MedicalHistoryCRUD()

    def _connect_redis(self) -> redis.Redis:
        """Kết nối với Redis sử dụng cấu hình từ biến môi trường."""
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

    def _retry_on_failure(self, func, *args, max_retries: int = 3, **kwargs):
        """Thử lại hàm nếu Redis lỗi."""
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except RedisConnectionError as e:
                logger.warning(f"Redis error on attempt {attempt + 1}: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error("Max retries reached, falling back to MongoDB")
                    return None
                self.redis = self._connect_redis()

    def _serialize_mongo_document(self, data: Dict) -> Dict:
        """Chuyển đổi các trường không JSON-serializable (như ObjectId) thành chuỗi."""
        if not data:
            return data
        serialized = data.copy()
        if '_id' in serialized:
            serialized['_id'] = str(serialized['_id'])
        return serialized

    # CRUD cho Patient
    def create_patient(self, patients: List[Dict]) -> List[str]:
        """
        Tạo bệnh nhân mới và cache vào Redis.

        Args:
            patients: Danh sách dữ liệu bệnh nhân.

        Returns:
            Danh sách user_id của các bệnh nhân đã tạo.
        """
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
                    'email': patient.get('email')
                }
                self._retry_on_failure(
                    self.redis.setex,
                    user_key,
                    timedelta(seconds=self.ttl),
                    json.dumps(cache_data)
                )
                self._retry_on_failure(
                    self.redis.setex,
                    patient_key,
                    timedelta(seconds=self.ttl),
                    json.dumps(cache_data)
                )
                logger.info(f"Cached patient {patient['username']} in Redis")
            return user_ids
        except Exception as e:
            logger.error(f"Error creating patients: {str(e)}")
            raise

    def get_patient_by_id(self, user_id: str) -> Optional[Dict]:
        """
        Lấy thông tin bệnh nhân theo user_id từ cache hoặc MongoDB.

        Args:
            user_id: ID của bệnh nhân.

        Returns:
            Thông tin bệnh nhân hoặc None nếu không tìm thấy.
        """
        patient_key = f"patient:{user_id}"
        cached_data = self._retry_on_failure(self.redis.get, patient_key)

        if cached_data:
            logger.info(f"Cache hit for patient {user_id}")
            self._retry_on_failure(self.redis.incr, f"access_count:{user_id}")
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
                'email': patient.get('email')
            }
            self._retry_on_failure(
                self.redis.setex,
                patient_key,
                timedelta(seconds=self.ttl),
                json.dumps(cache_data)
            )
            self._retry_on_failure(self.redis.incr, f"access_count:{user_id}")
            logger.info(f"Cached patient {user_id} from MongoDB")
            return cache_data
        return None

    def get_cached_user(self, username: str) -> Optional[Dict]:
        """
        Lấy thông tin bệnh nhân theo username từ cache hoặc MongoDB.

        Args:
            username: Tên người dùng.

        Returns:
            Thông tin bệnh nhân hoặc None nếu không tìm thấy.
        """
        user_key = f"user:{username}"
        cached_data = self._retry_on_failure(self.redis.get, user_key)

        if cached_data:
            logger.info(f"Cache hit for user {username}")
            return json.loads(cached_data)

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
                'email': patient.get('email')
            }
            self._retry_on_failure(
                self.redis.setex,
                user_key,
                timedelta(seconds=self.ttl),
                json.dumps(cache_data)
            )
            self._retry_on_failure(self.redis.incr, f"access_count:{patient['user_id']}")
            logger.info(f"Cached user {username} from MongoDB")
            return cache_data
        return None

    def search_patients(self, search_text: str, limit: int = 10, skip: int = 0) -> List[Dict]:
        """
        Tìm kiếm bệnh nhân theo tên hoặc username từ cache hoặc MongoDB.

        Args:
            search_text: Chuỗi tìm kiếm.
            limit: Số bản ghi tối đa (giới hạn tối đa 100).
            skip: Số bản ghi bỏ qua.

        Returns:
            Danh sách Bệnh nhân.
        """
        limit = min(limit, 100)
        cache_key = f"search:patients:{search_text}:{skip}:{limit}"
        cached_data = self._retry_on_failure(self.redis.get, cache_key)

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
                    'password': p.get('password'),
                    'name': p.get('name'),
                    'birthdate': p.get('birthdate'),
                    'gender': p.get('gender'),
                    'role': p.get('role'),
                    'work': p.get('work'),
                    'email': p.get('email')
                } for p in patients
            ]
            self._retry_on_failure(
                self.redis.setex,
                cache_key,
                timedelta(seconds=self.ttl),
                json.dumps(cache_data)
            )
            for p in patients:
                self._retry_on_failure(self.redis.incr, f"access_count:{p['user_id']}")
            logger.info(f"Cached search results for {search_text}")
            return cache_data
        return []

    def update_patient(self, user_id: str, patient_data: Dict) -> bool:
        """
        Cập nhật thông tin bệnh nhân và làm mới cache.

        Args:
            user_id: ID của bệnh nhân.
            patient_data: Dữ liệu cần cập nhật.

        Returns:
            True nếu cập nhật thành công.
        """
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
                    'email': patient.get('email')
                }
                self._retry_on_failure(
                    self.redis.setex,
                    f"patient:{user_id}",
                    timedelta(seconds=self.ttl),
                    json.dumps(cache_data)
                )
                self._retry_on_failure(
                    self.redis.setex,
                    f"user:{patient['username']}",
                    timedelta(seconds=self.ttl),
                    json.dumps(cache_data)
                )
                logger.info(f"Updated cache for patient {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating patient: {str(e)}")
            return False

    def delete_patient(self, user_id: str) -> bool:
        """
        Xóa bệnh nhân và làm mới cache.

        Args:
            user_id: ID của bệnh nhân.

        Returns:
            True nếu xóa thành công.
        """
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

    # CRUD cho MedicalHistory
    def create_medical_history(self, medical_history: Dict) -> bool:
        """
        Tạo lịch sử y tế mới và cache vào Redis.

        Args:
            medical_history: Dữ liệu lịch sử y tế.

        Returns:
            True nếu tạo thành công.
        """
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
            self._retry_on_failure(
                self.redis.setex,
                image_key,
                timedelta(seconds=self.ttl),
                json.dumps(serialized_history)
            )
            logger.info(f"Cached medical history for user_id {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error creating medical history: {str(e)}")
            return False

    def update_medical_history(self, user_id: str, medical_data: Dict) -> bool:
        """
        Cập nhật lịch sử y tế và làm mới cache.

        Args:
            user_id: ID của bệnh nhân.
            medical_data: Dữ liệu cần cập nhật.

        Returns:
            True nếu cập nhật thành công.
        """
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
                self._retry_on_failure(
                    self.redis.setex,
                    image_key,
                    timedelta(seconds=self.ttl),
                    json.dumps(serialized_data)
                )
                logger.info(f"Updated cache for medical history {user_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating medical history: {str(e)}")
            return False

    def get_medical_history(self, user_id: str) -> Optional[Dict]:
        """
        Lấy lịch sử y tế từ cache hoặc MongoDB.

        Args:
            user_id: ID của bệnh nhân.

        Returns:
            Bản ghi lịch sử y tế hoặc None.
        """
        image_key = f"medical:{user_id}:latest"
        cached_data = self._retry_on_failure(self.redis.get, image_key)

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
            self._retry_on_failure(
                self.redis.setex,
                image_key,
                timedelta(seconds=self.ttl),
                json.dumps(serialized_data)
            )
            logger.info(f"Cached medical history {user_id} from MongoDB")
            return serialized_data
        return None

    def delete_medical_history(self, user_id: str) -> bool:
        """
        Xóa lịch sử y tế và làm mới cache.

        Args:
            user_id: ID của bệnh nhân.

        Returns:
            True nếu xóa thành công.
        """
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

    # Cache từ Airflow
    def cache_from_airflow(self, combined_records: List[Dict], limit: int = 100) -> int:
        """
        Cache dữ liệu từ kết quả fetch_and_export_to_excel của Airflow.

        Args:
            combined_records: Danh sách bản ghi từ combined_df.
            limit: Số lượng bệnh nhân tối đa để cache.

        Returns:
            Số bản ghi được cache.
        """
        try:
            df = pd.DataFrame(combined_records)
            if df.empty:
                logger.info("No data to cache from Airflow")
                return 0

            access_counts = {}
            for _, record in df.iterrows():
                user_id = record.get('user_id')
                if user_id:
                    count = self._retry_on_failure(self.redis.get, f"access_count:{user_id}")
                    access_counts[user_id] = int(count or 0)

            if 'diagnosis_score' in df.columns:
                df['access_count'] = df['user_id'].map(access_counts).fillna(0)
                df = df.sort_values(by=['diagnosis_score', 'access_count'], ascending=[False, False]).head(limit)
            else:
                df = df.head(limit)

            cached_count = 0
            for _, record in df.iterrows():
                username = record.get('username')
                user_id = record.get('user_id')
                if not isinstance(username, str) or not username.strip() or not user_id:
                    logger.warning("Skipping record: missing or invalid username/user_id")
                    continue

                user_key = f"user:{username}"
                patient_key = f"patient:{user_id}"
                cache_data = {
                    'user_id': user_id,
                    'username': username,
                    'password': record.get('password'),
                    'name': record.get('name'),
                    'birthdate': record.get('birthdate'),
                    'gender': record.get('gender'),
                    'role': record.get('role'),
                    'work': record.get('work'),
                    'email': record.get('email')
                }
                self._retry_on_failure(
                    self.redis.setex,
                    user_key,
                    timedelta(seconds=self.ttl),
                    json.dumps(cache_data)
                )
                self._retry_on_failure(
                    self.redis.setex,
                    patient_key,
                    timedelta(seconds=self.ttl),
                    json.dumps(cache_data)
                )

                if 'image_id' in record and pd.notna(record['image_id']):
                    medical_data = {
                        'user_id': user_id,
                        'image_id': record.get('image_id'),
                        'image': record.get('image'),
                        'diagnosis_score': record.get('diagnosis_score'),
                        'date': [record.get('date')] if pd.notna(record.get('date')) else []
                    }
                    image_key = f"medical:{user_id}:latest"
                    serialized_medical = self._serialize_mongo_document(medical_data)
                    if 'date' in serialized_medical:
                        serialized_medical['date'] = [
                            d.isoformat() if isinstance(d, datetime) else d
                            for d in serialized_medical['date']
                        ]
                    self._retry_on_failure(
                        self.redis.setex,
                        image_key,
                        timedelta(seconds=self.ttl),
                        json.dumps(serialized_medical)
                    )
                cached_count += 1

            logger.info(f"Cached {cached_count} records from Airflow")
            return cached_count
        except Exception as e:
            logger.error(f"Error caching from Airflow: {str(e)}")
            return 0

    def invalidate_cache(self, prefix: str = "user:*") -> int:
        """
        Xóa cache theo prefix (user:*, patient:*, medical:*).

        Args:
            prefix: Mẫu key cần xóa.

        Returns:
            Số key đã xóa.
        """
        try:
            cursor = 0
            deleted_count = 0
            while True:
                cursor, keys = self.redis.scan(cursor, match=prefix, count=1000)
                if keys:
                    self.redis.delete(*keys)
                    deleted_count += len(keys)
                if cursor == 0:
                    break
            logger.info(f"Deleted {deleted_count} keys with prefix {prefix}")
            return deleted_count
        except Exception as e:
            logger.error(f"Error invalidating cache: {str(e)}")
            return 0

if __name__ == "__main__":
    # Dữ liệu mẫu
    unique_id = str(uuid.uuid4())[:8]
    sample_patient = {
        'user_id': f'test_{unique_id}',
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
        'user_id': f'test_{unique_id}',
        'image_id': f'img_{unique_id}',
        'image': 'https://example.com/image.jpg',
        'diagnosis_score': 0.85,
        'date': [],
        'comment': 'Test medical history'
    }
    sample_airflow_record = [{
        'user_id': f'test_{unique_id}',
        'username': f'testuser_{unique_id}',
        'name': 'Test User',
        'email': f'test_{unique_id}@example.com',
        'image_id': f'img_{unique_id}',
        'image': 'https://example.com/image.jpg',
        'diagnosis_score': 0.85,
        'date': datetime.now(timezone.utc).isoformat(),
        'role': 'patient',
        'gender': 'male',
        'birthdate': '01/01/1990'
    }]

    # Khởi tạo RedisCache
    redis_cache = None
    try:
        redis_cache = RedisCache(ttl_seconds=3600)
    except RedisConnectionError as e:
        print(f"Không thể kết nối Redis: {str(e)}. Thoát test.")
        exit(1)
    except Exception as e:
        print(f"Lỗi khởi tạo RedisCache: {str(e)}. Thoát test.")
        exit(1)

    # Hàm dọn dẹp dữ liệu sau test
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

    # Test case 1: Tạo bệnh nhân
    print("Test 1: Tạo bệnh nhân")
    try:
        user_ids = redis_cache.create_patient([sample_patient])
        assert user_ids == [sample_patient['user_id']], f"Expected [{sample_patient['user_id']}], got {user_ids}"
        cached = redis_cache.get_patient_by_id(sample_patient['user_id'])
        assert cached['username'] == sample_patient['username'], f"Expected username '{sample_patient['username']}', got {cached.get('username')}"
        print("Test 1 passed")
    except Exception as e:
        print(f"Test 1 failed: {str(e)}")
        cleanup(sample_patient['user_id'])
        exit(1)

    # Test case 2: Lấy bệnh nhân (cache hit)
    print("Test 2: Lấy bệnh nhân")
    try:
        patient = redis_cache.get_patient_by_id(sample_patient['user_id'])
        assert patient['username'] == sample_patient['username'], f"Expected username '{sample_patient['username']}', got {patient.get('username')}"
        print("Test 2 passed")
    except Exception as e:
        print(f"Test 2 failed: {str(e)}")
        cleanup(sample_patient['user_id'])
        exit(1)

    # Test case 3: Tìm kiếm bệnh nhân
    print("Test 3: Tìm kiếm bệnh nhân")
    try:
        patients = redis_cache.search_patients(sample_patient['username'], limit=10, skip=0)
        assert len(patients) == 1 and patients[0]['username'] == sample_patient['username'], f"Expected 1 patient with username '{sample_patient['username']}', got {patients}"
        cached = redis_cache.search_patients(sample_patient['username'], limit=10, skip=0)
        assert len(cached) == 1 and cached[0]['username'] == sample_patient['username'], f"Expected 1 cached patient, got {cached}"
        print("Test 3 passed")
    except Exception as e:
        print(f"Test 3 failed: {str(e)}")
        cleanup(sample_patient['user_id'])
        exit(1)

    # Test case 4: Cập nhật bệnh nhân
    print("Test 4: Cập nhật bệnh nhân")
    try:
        updated_data = {'name': 'Updated User'}  # Chỉ cập nhật name
        success = redis_cache.update_patient(sample_patient['user_id'], updated_data)
        assert success, "Update patient failed"
        patient = redis_cache.get_patient_by_id(sample_patient['user_id'])
        assert patient['name'] == 'Updated User', f"Expected name 'Updated User', got {patient.get('name')}"
        print("Test 4 passed")
    except Exception as e:
        print(f"Test 4 failed: {str(e)}")
        cleanup(sample_patient['user_id'])
        exit(1)

    # Test case 5: Tạo lịch sử y tế
    print("Test 5: Tạo lịch sử y tế")
    try:
        success = redis_cache.create_medical_history(sample_medical_history)
        assert success, "Create medical history failed"
        history = redis_cache.get_medical_history(sample_patient['user_id'])
        assert history['image_id'] == sample_medical_history['image_id'], f"Expected image_id '{sample_medical_history['image_id']}', got {history.get('image_id')}"
        print("Test 5 passed")
    except Exception as e:
        print(f"Test 5 failed: {str(e)}")
        cleanup(sample_patient['user_id'])
        exit(1)

    # Test case 6: Lấy lịch sử y tế
    print("Test 6: Lấy lịch sử y tế")
    try:
        history = redis_cache.get_medical_history(sample_patient['user_id'])
        assert history['image_id'] == sample_medical_history['image_id'], f"Expected image_id '{sample_medical_history['image_id']}', got {history.get('image_id')}"
        print("Test 6 passed")
    except Exception as e:
        print(f"Test 6 failed: {str(e)}")
        cleanup(sample_patient['user_id'])
        exit(1)

    # Test case 7: Cập nhật lịch sử y tế
    print("Test 7: Cập nhật lịch sử y tế")
    try:
        updated_medical = sample_medical_history.copy()
        updated_medical['diagnosis_score'] = 0.9
        success = redis_cache.update_medical_history(sample_patient['user_id'], updated_medical)
        assert success, "Update medical history failed"
        history = redis_cache.get_medical_history(sample_patient['user_id'])
        assert history['diagnosis_score'] == 0.9, f"Expected diagnosis_score 0.9, got {history.get('diagnosis_score')}"
        print("Test 7 passed")
    except Exception as e:
        print(f"Test 7 failed: {str(e)}")
        cleanup(sample_patient['user_id'])
        exit(1)

    # Test case 8: Cache từ Airflow
    print("Test 8: Cache từ Airflow")
    try:
        count = redis_cache.cache_from_airflow(sample_airflow_record, limit=1)
        assert count == 1, f"Expected 1 record cached, got {count}"
        patient = redis_cache.get_patient_by_id(sample_patient['user_id'])
        assert patient['username'] == sample_patient['username'], f"Expected username '{sample_patient['username']}', got {patient.get('username')}"
        print("Test 8 passed")
    except Exception as e:
        print(f"Test 8 failed: {str(e)}")
        cleanup(sample_patient['user_id'])
        exit(1)

    # Test case 9: Xóa lịch sử y tế
    print("Test 9: Xóa lịch sử y tế")
    try:
        success = redis_cache.delete_medical_history(sample_patient['user_id'])
        assert success, "Delete medical history failed"
        history = redis_cache.get_medical_history(sample_patient['user_id'])
        assert history is None, f"Expected None, got {history}"
        print("Test 9 passed")
    except Exception as e:
        print(f"Test 9 failed: {str(e)}")
        cleanup(sample_patient['user_id'])
        exit(1)

    # Test case 10: Xóa bệnh nhân
    print("Test 10: Xóa bệnh nhân")
    try:
        success = redis_cache.delete_patient(sample_patient['user_id'])
        assert success, "Delete patient failed"
        patient = redis_cache.get_patient_by_id(sample_patient['user_id'])
        assert patient is None, f"Expected None, got {patient}"
        print("Test 10 passed")
    except Exception as e:
        print(f"Test 10 failed: {str(e)}")
        cleanup(sample_patient['user_id'])
        exit(1)

    # Test case 11: Xóa cache theo prefix
    print("Test 11: Xóa cache theo prefix")
    try:
        deleted_count = redis_cache.invalidate_cache('user:*')
        assert deleted_count >= 0, f"Expected non-negative deleted count, got {deleted_count}"
        patient = redis_cache.get_patient_by_id(sample_patient['user_id'])
        assert patient is None, f"Expected None, got {patient}"
        print("Test 11 passed")
    except Exception as e:
        print(f"Test 11 failed: {str(e)}")
        cleanup(sample_patient['user_id'])
        exit(1)

    # Test case 12: Tạo bệnh nhân với email không hợp lệ
    print("Test 12: Tạo bệnh nhân với email không hợp lệ")
    try:
        invalid_patient = sample_patient.copy()
        invalid_patient['user_id'] = f'invalid_{unique_id}'
        invalid_patient['username'] = f'invaliduser_{unique_id}'
        invalid_patient['email'] = 'invalid_email'
        user_ids = redis_cache.create_patient([invalid_patient])
        print("Test 12 failed: Expected ValidationError for invalid email")
        cleanup(invalid_patient['user_id'])
        exit(1)
    except Exception as e:
        assert "ValidationError" in str(type(e)), f"Expected ValidationError, got {str(e)}"
        print("Test 12 passed")
        cleanup(f'invalid_{unique_id}')

    # Test case 13: Tạo bệnh nhân với username đã tồn tại
    print("Test 13: Tạo bệnh nhân với username đã tồn tại")
    try:
        duplicate_patient = sample_patient.copy()
        duplicate_patient['user_id'] = f'duplicate_{unique_id}'
        duplicate_patient['email'] = f'duplicate_{unique_id}@example.com'
        user_ids = redis_cache.create_patient([duplicate_patient])
        print("Test 13 failed: Expected ValidationError for duplicate username")
        cleanup(duplicate_patient['user_id'])
        exit(1)
    except Exception as e:
        assert "ValidationError" in str(type(e)), f"Expected ValidationError, got {str(e)}"
        print("Test 13 passed")
        cleanup(f'duplicate_{unique_id}')

    # Test case 14: Cập nhật lịch sử y tế với diagnosis_score không hợp lệ
    print("Test 14: Cập nhật lịch sử y tế với diagnosis_score không hợp lệ")
    try:
        # Tạo lại lịch sử y tế để test
        redis_cache.create_medical_history(sample_medical_history)
        invalid_medical = sample_medical_history.copy()
        invalid_medical['diagnosis_score'] = 1.5  # Ngoài khoảng [0, 1]
        success = redis_cache.update_medical_history(sample_patient['user_id'], invalid_medical)
        assert not success, "Expected update to fail for invalid diagnosis_score"
        print("Test 14 passed")
    except Exception as e:
        print(f"Test 14 failed: {str(e)}")
        cleanup(sample_patient['user_id'])
        exit(1)

    # Test case 15: Tìm kiếm bệnh nhân với chuỗi rỗng
    print("Test 15: Tìm kiếm bệnh nhân với chuỗi rỗng")
    try:
        patients = redis_cache.search_patients("", limit=10, skip=0)
        assert len(patients) >= 1, f"Expected at least 1 patient, got {len(patients)}"
        print("Test 15 passed")
    except Exception as e:
        print(f"Test 15 failed: {str(e)}")
        cleanup(sample_patient['user_id'])
        exit(1)

    # Test case 16: Tạo nhiều bệnh nhân cùng lúc
    print("Test 16: Tạo nhiều bệnh nhân cùng lúc")
    try:
        multiple_patients = [
            {
                'user_id': f'bulk_{unique_id}_{i}',
                'username': f'bulkuser_{unique_id}_{i}',
                'password': 'testpassword',
                'name': f'Bulk User {i}',
                'birthdate': '01/01/1990',
                'gender': 'male',
                'role': 'patient',
                'work': 'Engineer',
                'email': f'bulk_{unique_id}_{i}@example.com'
            } for i in range(5)
        ]
        user_ids = redis_cache.create_patient(multiple_patients)
        assert len(user_ids) == 5, f"Expected 5 user_ids, got {len(user_ids)}"
        for patient in multiple_patients:
            cached = redis_cache.get_patient_by_id(patient['user_id'])
            assert cached['username'] == patient['username'], f"Expected username '{patient['username']}', got {cached.get('username')}"
        print("Test 16 passed")
        # Dọn dẹp các bệnh nhân vừa tạo
        for patient in multiple_patients:
            cleanup(patient['user_id'])
    except Exception as e:
        print(f"Test 16 failed: {str(e)}")
        for patient in multiple_patients:
            cleanup(patient['user_id'])
        cleanup(sample_patient['user_id'])
        exit(1)

    # Test case 17: Xóa bệnh nhân không tồn tại
    print("Test 17: Xóa bệnh nhân không tồn tại")
    try:
        non_existent_id = f'nonexistent_{unique_id}'
        success = redis_cache.delete_patient(non_existent_id)
        assert not success, "Expected delete to fail for non-existent patient"
        print("Test 17 passed")
    except Exception as e:
        print(f"Test 17 failed: {str(e)}")
        cleanup(sample_patient['user_id'])
        exit(1)

    # Dọn dẹp cuối cùng
    cleanup(sample_patient['user_id'])
    print("All tests completed!")