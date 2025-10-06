from fastapi import APIRouter, HTTPException
from typing import List
from pydantic import BaseModel
import json
import os

router = APIRouter()

# Load papers
DB_FILE = os.path.join(os.path.dirname(__file__), "json_db.json")
if not os.path.exists(DB_FILE):
    sample_papers = [
        {"id": "paper1", "title": "Example Paper 1", "summary": "Summary of paper 1", "year": "2025", "pdf_link": "http://example.com/paper1.pdf"}
    ]
    with open(DB_FILE, "w", encoding="utf-8") as f:
        json.dump(sample_papers, f, indent=4)

with open(DB_FILE, "r", encoding="utf-8") as f:
    all_papers = json.load(f)

class Paper(BaseModel):
    id: str
    title: str
    summary: str
    year: str
    pdf_link: str

user_libraries = {}

def find_paper_by_id(paper_id: str):
    for paper in all_papers:
        if paper["id"] == paper_id:
            return paper
    return None

@router.get("/")
def library_root():
    return {"message": "Library API is working!"}

@router.get("/{user_id}", response_model=List[Paper])
def get_library(user_id: str):
    return user_libraries.get(user_id, [])

@router.post("/{user_id}/{paper_id}")
def add_to_library(user_id: str, paper_id: str):
    paper = find_paper_by_id(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    if user_id not in user_libraries:
        user_libraries[user_id] = []
    if any(p["id"] == paper["id"] for p in user_libraries[user_id]):
        raise HTTPException(status_code=400, detail="Paper already in library")
    user_libraries[user_id].append(paper)
    return {"message": f"Paper '{paper['title']}' added successfully"}

@router.delete("/{user_id}/{paper_id}")
def remove_from_library(user_id: str, paper_id: str):
    if user_id not in user_libraries:
        raise HTTPException(status_code=404, detail="Library not found")
    user_libraries[user_id] = [p for p in user_libraries[user_id] if p["id"] != paper_id]
    return {"message": "Paper removed successfully"}
