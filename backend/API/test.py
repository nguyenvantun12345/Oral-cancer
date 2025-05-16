import bcrypt
import logging

logger = logging.getLogger(__name__) # Giả sử bạn đã cấu hình logger

# Mật khẩu bạn muốn hash
plain_password_to_hash = '123456'

# Tạo salt và hash mật khẩu
# Lưu ý: mỗi lần chạy, salt sẽ khác nhau dẫn đến hash khác nhau
def hash_password(password: str) -> str:
    """Hash mật khẩu."""
    try:
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        logger.info("Password hashed successfully")
        return hashed
    except Exception as e:
        logger.error(f"Error hashing password: {str(e)}")
        raise
generated_hashed_password = hash_password(plain_password_to_hash)

# Bây giờ, nếu bạn sử dụng generated_hashed_password này với hàm verify_password, nó sẽ trả về True
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Xác minh mật khẩu."""
    try:
        result = bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
        logger.info("Password verification completed")
        return result
    except Exception as e:
        logger.error(f"Error verifying password: {str(e)}")
        return False

# Kiểm tra với hash vừa tạo
is_correct_new = verify_password(plain_password_to_hash, generated_hashed_password)
print(f"Xác minh '{plain_password_to_hash}' với hash vừa tạo ({generated_hashed_password}): {is_correct_new}") # Sẽ là True

# Kiểm tra với hash bạn cung cấp ban đầu
provided_hashed_password = '$2b$12$qzvgnLg0B.KjD4PvuXN.3.Rzk4Z.vuJL95VhpnxU5uIOMPMSo.1FS'
is_correct_original = verify_password(plain_password_to_hash, provided_hashed_password)
print(f"Xác minh '{plain_password_to_hash}' với hash bạn cung cấp ({provided_hashed_password}): {is_correct_original}") # Sẽ là False