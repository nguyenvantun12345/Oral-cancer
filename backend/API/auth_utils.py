import bcrypt
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def hash_password(password: str) -> str:
    """Hash mật khẩu."""
    try:
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        logger.info("Password hashed successfully")
        return hashed
    except Exception as e:
        logger.error(f"Error hashing password: {str(e)}")
        raise

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Xác minh mật khẩu."""
    try:
        logger.info(f"Verifying password. Plain password length: {len(plain_password)}")
        logger.info(f"Hashed password length: {len(hashed_password)}")
        logger.info(f"Hashed password is: {hashed_password}...")
        result = bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
        logger.info(f"Password verification result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error verifying password: {str(e)}")
        logger.error(f"Plain password length: {len(plain_password)}")
        logger.error(f"Hashed password length: {len(hashed_password)}")
        return False
    