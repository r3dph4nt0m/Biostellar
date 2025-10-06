from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .features.my_library import router as library_router

app = FastAPI(title="NASA Research Explorer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use include_router, not mount
app.include_router(library_router, prefix="/library")
