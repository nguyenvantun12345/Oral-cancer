from pymongo import MongoClient
from pymongo.server_api import ServerApi
from pymongo.errors import ServerSelectionTimeoutError, OperationFailure
import logging

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mongodb_connection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MongoDB_Connection")

class MongoDBConnection:
    _instance = None
    _uri = "mongodb+srv://havanduy1412:fQ47O2571BDtmMrv@duymongodb.tqkn3.mongodb.net/?retryWrites=true&w=majority&appName=DUYMONGODB"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MongoDBConnection, cls).__new__(cls)
            cls._instance._connect()
        return cls._instance

    def _connect(self):
        try:
            logger.info("Đang thử kết nối với MongoDB...")
            self.client = MongoClient(
                self._uri,
                server_api=ServerApi('1'),
                maxPoolSize=50,
                minPoolSize=5,
                waitQueueTimeoutMS=6000,
                connectTimeoutMS=5000,
                serverSelectionTimeoutMS=5000
            )
            self.client.admin.command('ping')
            self.db = self.client["MyProject"]
            logger.info("✅ Kết nối thành công với MongoDB")
        except (ServerSelectionTimeoutError, OperationFailure) as e:
            logger.error(f"❌ Không thể kết nối với MongoDB: {str(e)}")
            raise ConnectionError("Không thể kết nối với MongoDB")

    def check_and_reconnect(self):
        try:
            self.client.admin.command('ping')
            return True
        except Exception as e:
            logger.warning(f"Phát hiện lỗi kết nối: {str(e)}. Đang thử kết nối lại...")
            self._connect()
            return True

    def check_database_size(self, threshold_gb=490):
        try:
            stats = self.db.command("dbStats")
            size_gb = stats["dataSize"] / (1024 * 1024 * 1024)
            if size_gb > threshold_gb:
                logger.warning(f"Cơ sở dữ liệu đã đạt {size_gb:.2f}GB, vượt ngưỡng {threshold_gb}GB.")
                return False
            return True
        except Exception as e:
            logger.error(f"Lỗi khi kiểm tra kích thước cơ sở dữ liệu: {str(e)}")
            return self.check_and_reconnect()

def get_database():
    connection = MongoDBConnection()
    connection.check_and_reconnect()
    connection.check_database_size()
    return connection.db

if __name__ == "__main__":
    db = get_database()
    print("Connected to database:", db.name)
    print("Collections:", db.list_collection_names())