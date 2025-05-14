import random
import string
import smtplib
from email.mime.text import MIMEText
from twilio.rest import Client
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_otp(length: int = 6) -> str:
    """Tạo OTP ngẫu nhiên."""
    characters = string.digits
    return ''.join(random.choice(characters) for _ in range(length))

def send_email_otp(email: str, otp: str):
    """Gửi OTP qua email."""
    try:
        smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", 587))
        smtp_user = os.getenv("SMTP_USER")
        smtp_password = os.getenv("SMTP_PASSWORD")
        
        if not smtp_user or not smtp_password:
            logger.error("SMTP credentials not configured")
            raise Exception("SMTP credentials not configured")
        
        msg = MIMEText(f"Your OTP for login is: {otp}\nThis OTP is valid for 5 minutes.")
        msg['Subject'] = "Your Login OTP"
        msg['From'] = smtp_user
        msg['To'] = email
        
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        
        logger.info(f"OTP sent to email: {email}")
    except Exception as e:
        logger.error(f"Failed to send OTP to {email}: {str(e)}")
        raise

def send_sms_otp(phone: str, otp: str):
    """Gửi OTP qua SMS sử dụng Twilio."""
    try:
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        from_number = os.getenv("TWILIO_PHONE_NUMBER")
        
        if not account_sid or not auth_token or not from_number:
            logger.error("Twilio credentials not configured")
            raise Exception("Twilio credentials not configured")
        
        client = Client(account_sid, auth_token)
        message = client.messages.create(
            body=f"Your OTP for login is: {otp}\nThis OTP is valid for 5 minutes.",
            from_=from_number,
            to=phone
        )
        
        logger.info(f"OTP sent to phone: {phone}, Message SID: {message.sid}")
    except Exception as e:
        logger.error(f"Failed to send OTP to {phone}: {str(e)}")
        raise