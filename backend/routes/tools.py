from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from tools.emi_calculator import calculate_emi

router = APIRouter(prefix="/api/tools", tags=["tools"])

class EMIRequest(BaseModel):
    principal: float
    annual_rate: float
    months: int

class EMIResponse(BaseModel):
    emi: float
    total_payment: float
    total_interest: float

@router.post("/emi", response_model=EMIResponse)
def get_emi(req: EMIRequest):
    try:
        emi = calculate_emi(req.principal, req.annual_rate, req.months)
        total_payment = round(emi * req.months, 2)
        total_interest = round(total_payment - req.principal, 2)
        return EMIResponse(
            emi=emi,
            total_payment=total_payment,
            total_interest=total_interest
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")
