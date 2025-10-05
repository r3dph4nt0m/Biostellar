from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from my_library import app as library_app

app = FastAPI(title="NASA Research Explorer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/library", library_app)
