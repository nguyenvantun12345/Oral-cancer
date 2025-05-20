from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os
import logging
from fastapi.middleware.cors import CORSMiddleware
# Load environment variables from .env file
load_dotenv()

# Verify JWT_SECRET is loaded
if not os.getenv("JWT_SECRET"):
    logging.warning("JWT_SECRET not set in environment variables during main.py initialization")

from api import router as api_router
from patient import router as patient_router
from admin import router as admin_router
from diagnosis import router as diagnosis_router

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # or ["*"] to allow all (not recommended for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    error_messages = [err['msg'] for err in exc.errors() if err['msg']]
    if any("birthdate must be in format dd/mm/yyyy" in msg for msg in error_messages):
        error_messages = ["birthdate must be in format dd/mm/yyyy"]
    return JSONResponse(
        status_code=400,
        content={"detail": error_messages}
    )

app.include_router(api_router)
app.include_router(patient_router)
app.include_router(admin_router)
app.include_router(diagnosis_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)