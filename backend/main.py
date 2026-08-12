from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.routes import prediction, rag, tools, neurobot, documents

app = FastAPI(
    title="NeuroFinance AI API",
    description="Decision Intelligence Platform with RAG and Agentic AI",
    version="0.1.0"
)

# Set up CORS middleware to allow connections from React dashboard (Vite default is 5173, Create-React-App is 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount temporary upload folder static assets for front-end previewing
app.mount("/static", StaticFiles(directory="data/temp_uploads"), name="static")

# Register routers
app.include_router(prediction.router)
app.include_router(rag.router)
app.include_router(tools.router)
app.include_router(neurobot.router)
app.include_router(documents.router)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "neurofinance-api",
        "version": "0.1.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
