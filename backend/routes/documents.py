from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import shutil
import uuid
from pathlib import Path
import pandas as pd
import re
import numpy as np

from document_intelligence.predict import DocumentIntelligencePredictor
from backend.routes.prediction import get_raw_data, shap_service, RiskReportResponse

router = APIRouter(prefix="/api/documents", tags=["documents"])

# Create temporary upload folder
UPLOAD_DIR = Path("data/temp_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Lazy initialized predictor
predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        model_path = Path("models/document_cnn.pt")
        predictor = DocumentIntelligencePredictor(model_path=str(model_path))
    return predictor

class DocumentAnalyzeRequest(BaseModel):
    customer_id: int
    filename: str

@router.post("/classify")
async def classify_document(file: UploadFile = File(...)):
    """
    Saves uploaded file to disk and runs CNN classification.
    Returns classified document type, confidence score, and top-3 predictions.
    """
    # Verify file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files (.jpg, .jpeg, .png) are supported.")
        
    # Save file with a unique name to allow concurrent previews and keep filename keyword hints
    ext = Path(file.filename).suffix
    if not ext:
        ext = ".jpg"
    original_stem = Path(file.filename).stem
    # Replace non-alphanumeric characters with underscores
    clean_stem = re.sub(r'[^a-zA-Z0-9_-]', '_', original_stem)
    unique_filename = f"{clean_stem}_{uuid.uuid4().hex[:8]}{ext}"
    dest_path = UPLOAD_DIR / unique_filename
    
    try:
        with open(dest_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
        
    # Run CNN + OCR Inference
    try:
        pred_service = get_predictor()
        res = pred_service.predict(str(dest_path))
        
        # Add the unique filename to response so frontend can link it
        res["filename"] = unique_filename
        return res
    except Exception as e:
        # Clean up file on failure
        if dest_path.exists():
            dest_path.unlink()
        raise HTTPException(status_code=500, detail=f"Classification error: {e}")

@router.post("/analyze")
def analyze_document_risk(req: DocumentAnalyzeRequest):
    """
    Runs end-to-end integration:
    Reads document -> extracts fields -> modifies customer profile -> runs FT-Transformer risk prediction -> SHAP attributions.
    """
    global shap_service
    
    # 1. Verify file exists
    file_path = UPLOAD_DIR / req.filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Uploaded file not found. Upload it first.")
        
    # 2. Run predictor
    try:
        pred_service = get_predictor()
        doc_res = pred_service.predict(str(file_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
        
    # 3. Load baseline customer record
    df = get_raw_data()
    if df.empty:
        raise HTTPException(status_code=503, detail="Main dataset not loaded.")
        
    customer_record = df[df["SK_ID_CURR"] == req.customer_id].copy()
    if customer_record.empty:
        raise HTTPException(status_code=404, detail=f"Customer ID {req.customer_id} not found in database.")
        
    # 4. Integrate extracted fields into the customer record
    extracted = doc_res["extracted_fields"]
    warnings = doc_res["validation_warnings"]
    
    if doc_res["document_type"] == "PAYSLIP" and "monthly_income" in extracted and extracted["monthly_income"]:
        # Update annual income (monthly * 12)
        annual_income = extracted["monthly_income"] * 12
        customer_record.loc[:, "AMT_INCOME_TOTAL"] = annual_income
        print(f"Document Integration: Updating AMT_INCOME_TOTAL for client {req.customer_id} to annual salary {annual_income}")
    elif doc_res["document_type"] == "BANK_STATEMENT" and "average_balance" in extracted and extracted["average_balance"]:
        # We can trigger dynamic alerts if bank statement shows delinquency warnings
        pass
        
    # 5. Run prediction and SHAP explanation on the modified record
    from backend.routes.prediction import get_shap_service
    active_shap_service = get_shap_service()
    
    try:
        report = active_shap_service.explain(customer_record.drop(columns=["TARGET"], errors="ignore"))
        
        # Parse profile for return
        profile_dict = {}
        for col in customer_record.columns:
            val = customer_record.iloc[0][col]
            if pd.isna(val):
                profile_dict[col] = None
            elif isinstance(val, (np.integer, int)):
                profile_dict[col] = int(val)
            elif isinstance(val, (np.floating, float)):
                profile_dict[col] = float(val)
            else:
                profile_dict[col] = str(val)
                
        return {
            "document_type": doc_res["document_type"],
            "confidence": doc_res["confidence"],
            "extracted_fields": extracted,
            "validation_warnings": warnings,
            "raw_text": doc_res["raw_text"],
            "risk_report": {
                "default_probability": report["default_probability"],
                "risk_category": report["risk_category"],
                "financial_health_score": report["financial_health_score"],
                "attributions": report["attributions"],
                "profile": profile_dict
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Credit Risk simulation failed: {e}")
