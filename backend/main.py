from fastapi import FastAPI
from API import image, patient, auth  # Add each router as needed

app = FastAPI()

# Register routers
app.include_router(image.router, prefix="/api", tags=["Image"])
app.include_router(patient.router, prefix="/api", tags=["Patient"])
app.include_router(auth.router, prefix="/api", tags=["Auth"])