from fastapi import FastAPI, HTTPException
from typing import List
from pydantic import BaseModel
import json

app = FastAPI()

with open("json_db.json", "r", encoding="utf-8") as f:
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

@app.get("/library/{user_id}", response_model=List[Paper])
def get_library(user_id: str):
    return user_libraries.get(user_id, [])

@app.post("/library/{user_id}/{paper_id}")
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

@app.delete("/library/{user_id}/{paper_id}")
def remove_from_library(user_id: str, paper_id: str):
    if user_id not in user_libraries:
        raise HTTPException(status_code=404, detail="Library not found")
    
    user_libraries[user_id] = [p for p in user_libraries[user_id] if p["id"] != paper_id]
    return {"message": "Paper removed successfully"}
