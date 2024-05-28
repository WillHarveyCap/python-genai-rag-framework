from fastapi import FastAPI
from src.routes import search, upload_file, classification
import uvicorn

app = FastAPI(debug=True)
app.include_router(search.router)
app.include_router(upload_file.router)
app.include_router(classification.router)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8081)