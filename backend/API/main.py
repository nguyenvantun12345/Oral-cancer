from fastapi import FastAPI
import image, patient, auth

app = FastAPI()

# Register routers
app.include_router(image.router, prefix="/image", tags=["Image"])
app.include_router(patient.router, prefix="/patient", tags=["Patient"])
app.include_router(auth.router, tags=["Auth"])