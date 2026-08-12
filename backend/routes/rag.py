from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from rag.rag_service import NeuroFinanceRAGService

router = APIRouter(prefix="/api/rag", tags=["rag"])

# Initialize RAG Service
try:
    rag_service = NeuroFinanceRAGService()
except Exception as e:
    print(f"RAG API Router Warning: Could not initialize RAG Service: {e}")
    rag_service = None

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

@router.post("/query", response_model=QueryResponse)
def run_query(req: QueryRequest):
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG Service is not available.")
    try:
        res = rag_service.query(req.query)
        return QueryResponse(
            answer=res["answer"],
            sources=res["sources"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
