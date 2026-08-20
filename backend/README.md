# 🚀 AI Data Readiness Platform - Backend

FastAPI backend providing data diagnosis, health scoring, explainable remediation recommendations, and candidate model benchmarking.

## 🛠️ Quick Setup

1. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the API server:**
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

4. **Access the API Documentation:**
   - Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
   - ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)
   - Health Check: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)
