# Biostellar
Unlocking 60 years of Space Biology

# 🛰️ BioStellar: Space Biology Knowledge Engine

**Goal:** Unlock 60 years of NASA bioscience research to help scientists and the public explore how life thrives beyond Earth.

## 👩‍🚀 Features
- Search 600+ space biology studies by organism, mission, and experiment type  
- AI chatbot "Archivist" that answers science & public questions  
- Interactive dashboards showing knowledge gaps  
- Mission Mode: story-driven summaries of discoveries & gaps

## 🗂️ Structure
- `data/` → raw NASA study links, AI-processed summaries, and demo datasets  
- `backend/` → API and database handling  
- `frontend/` → web interface and visualizations  
- `ai/` → chatbot logic, prompts, and mission briefs  

## 🚀 Getting Started
1. Clone the repo  
   ```bash
   git clone https://github.com/r3dph4nt0m/biostellar.git
   ```

2. Set up environment
   ```bash
   cd biostellar
   copy .env.example .env
   # Add your CEREBRAS_API_KEY to .env
   ```

3. Start with Docker
   ```bash
   docker-compose up --build
   ```

4. Access the application
   - Web UI: http://localhost:3000
   - API: http://localhost:8000
   - Docs: http://localhost:8000/docs

## 🛠️ Tech Stack
- **Backend**: FastAPI + WebSockets
- **Frontend**: React 18
- **AI**: ChatCerebras
- **Infrastructure**: Docker + Compose

## 🧪 Development
1. Install backend dependencies
   ```bash
   cd ai
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Install frontend dependencies
   ```bash
   cd frontend
   npm install
   ```

3. Run locally
   ```bash
   # Terminal 1 - Backend
   cd ai
   uvicorn main:app --reload

   # Terminal 2 - Frontend
   cd frontend
   npm start
   ```

## 🤝 Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License
MIT License - see [LICENSE](LICENSE)