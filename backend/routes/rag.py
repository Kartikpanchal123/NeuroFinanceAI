from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from rag.rag_service import NeuroFinanceRAGService

router = APIRouter(prefix="/api/rag", tags=["rag"])

# Initialize RAG Service
rag_service = None

def get_rag_service():
    global rag_service
    if rag_service is None:
        try:
            print("RAG API Router: Initializing RAG Service...")
            rag_service = NeuroFinanceRAGService()
        except Exception as e:
            print(f"RAG API Router Warning: Could not initialize RAG Service: {e}")
            rag_service = None
    return rag_service

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

@router.post("/query", response_model=QueryResponse)
def run_query(req: QueryRequest):
    service = get_rag_service()
    if not service:
        raise HTTPException(status_code=503, detail="RAG Service is not available.")
    try:
        res = service.query(req.query)
        return QueryResponse(
            answer=res["answer"],
            sources=res["sources"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
