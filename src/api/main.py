from fastapi import FastAPI
from src.api.routes import router

app = FastAPI(
    title="3D ULPIN API",
    description="3D building, roof and solar information API",
    version="1.0.0"
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "status": "running",
        "message": "3D ULPIN API is working"
    }