import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SECRET_KEY = os.getenv("JWT_SECRET", "f7b3e8c2a9d4f1e6b0c7a8d5e2f3b9c0a1d4e7f8")
if not os.getenv("JWT_SECRET"):
    logger.warning("JWT_SECRET not set in environment variables, using default key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30