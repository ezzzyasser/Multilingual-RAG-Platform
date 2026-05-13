# RAG API (Phase 3 Bonus)


## Test the Endpoints

Open your browser:
- **Swagger UI (interactive docs):** http://127.0.0.1:8000/docs
- **Health check:** http://127.0.0.1:8000/health

Or use the included `test_api.http` file with the REST Client VS Code extension.

---

## API Endpoints Summary

| Method | Endpoint | What it does |
|--------|----------|--------------|
| GET | `/health` | System health check |
| POST | `/ask-question` | Ask a question, get an answer |
| POST | `/evaluate` | Run evaluation metrics |
| GET | `/analytics` | View query stats from the DB |
