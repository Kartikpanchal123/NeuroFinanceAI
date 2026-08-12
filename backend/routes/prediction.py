from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from pathlib import Path

router = APIRouter(prefix="/api/prediction", tags=["prediction"])

import os

class MockSHAPService:
    def explain(self, raw_customer_df, top_k=5):
        try:
            ext2 = float(raw_customer_df.get("EXT_SOURCE_2", pd.Series([0.5])).iloc[0])
            ext3 = float(raw_customer_df.get("EXT_SOURCE_3", pd.Series([0.5])).iloc[0])
        except Exception:
            ext2 = 0.5
            ext3 = 0.5
            
        if pd.isna(ext2): ext2 = 0.5
        if pd.isna(ext3): ext3 = 0.5
        
        prob = 0.45 - 0.25 * ext2 - 0.15 * ext3
        prob = max(0.01, min(0.99, prob))
        prob = round(float(prob), 4)
        
        risk_category = "Low"
        if prob > 0.35:
            risk_category = "High"
        elif prob > 0.15:
            risk_category = "Medium"
            
        health_score = round(100.0 - (prob * 100.0), 2)
        
        top_risk_factors = [
            {"feature": "External Credit Score (Source 3)", "value": round(float(0.15 * (1.0 - ext3)), 4)},
            {"feature": "Loan Annuity (Monthly Payment)", "value": 0.08},
            {"feature": "Vehicle / Car Ownership", "value": 0.04}
        ]
        top_saving_factors = [
            {"feature": "Annual Income", "value": -0.12},
            {"feature": "External Credit Score (Source 2)", "value": round(float(-0.18 * ext2), 4)}
        ]
        
        return {
            "default_probability": prob,
            "risk_category": risk_category,
            "financial_health_score": health_score,
            "method": "Lightweight-Decision-Mock",
            "attributions": {
                "top_risk_factors": top_risk_factors,
                "top_saving_factors": top_saving_factors
            }
        }

def get_shap_service():
    global shap_service
    if shap_service is None:
        if os.environ.get("RENDER") is not None or os.environ.get("DISABLE_PYTORCH") == "true":
            print("Prediction API Router: Running on Render (memory-constrained). Using MockSHAPService.")
            shap_service = MockSHAPService()
        else:
            project_root = Path(__file__).resolve().parent.parent.parent
            model_path = project_root / "models" / "ft_transformer.pt"
            prep_path = project_root / "models" / "preprocessor.pkl"
            if model_path.exists() and prep_path.exists():
                try:
                    from explainability.shap_service import NeuroFinanceSHAPService
                    shap_service = NeuroFinanceSHAPService()
                except Exception as e:
                    print(f"Failed to load NeuroFinanceSHAPService: {e}. Falling back to Mock.")
                    shap_service = MockSHAPService()
            else:
                print("Model weights not found. Using MockSHAPService.")
                shap_service = MockSHAPService()
    return shap_service

# Initialize SHAP service and load dataset for lookups
shap_service = None
raw_data = None

def get_raw_data():
    global raw_data
    if raw_data is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        raw_path = project_root / "data" / "raw" / "application_train.csv"
        backup_path = project_root / "models" / "sample_profiles.csv"
        if raw_path.exists():
            print("Prediction API Router: Loading application_train.csv for lookups...")
            raw_data = pd.read_csv(raw_path, nrows=10000)
            print(f"Prediction API Router: Loaded {len(raw_data)} customer profiles.")
        elif backup_path.exists():
            print("Prediction API Router: Loading backup sample_profiles.csv for lookups...")
            raw_data = pd.read_csv(backup_path)
            print(f"Prediction API Router: Loaded {len(raw_data)} backup customer profiles.")
        else:
            print("Prediction API Router Warning: No profile datasets found.")
            raw_data = pd.DataFrame()
    return raw_data

class CustomPredictRequest(BaseModel):
    features: Dict[str, Any]

class RiskReportResponse(BaseModel):
    sk_id_curr: int
    default_probability: float
    risk_category: str
    financial_health_score: float
    attributions: Dict[str, Any]
    profile: Dict[str, Any]

class CustomerListItem(BaseModel):
    sk_id_curr: int
    target: int
    gender: str
    income: float
    credit: float

@router.get("/samples", response_model=List[CustomerListItem])
def get_sample_customers(limit: int = 50):
    """Returns a list of sample customer IDs and key profile metrics for the frontend selection."""
    df = get_raw_data()
    if df.empty:
        return []
    
    # Take a mix of default and non-default cases for interesting demonstrations
    df_defaults = df[df["TARGET"] == 1].head(limit // 2)
    df_repays = df[df["TARGET"] == 0].head(limit - len(df_defaults))
    sample_df = pd.concat([df_defaults, df_repays])
    
    samples = []
    for _, row in sample_df.iterrows():
        samples.append(CustomerListItem(
            sk_id_curr=int(row["SK_ID_CURR"]),
            target=int(row["TARGET"]),
            gender=str(row.get("CODE_GENDER", "Unknown")),
            income=float(row.get("AMT_INCOME_TOTAL", 0.0)),
            credit=float(row.get("AMT_CREDIT", 0.0))
        ))
    return samples

@router.get("/customer/{sk_id}", response_model=RiskReportResponse)
def predict_customer(sk_id: int):
    global shap_service
    # Lazy load SHAP service if model trained in the background
    if shap_service is None:
        shap_service = get_shap_service()

    df = get_raw_data()
    if df.empty:
        raise HTTPException(status_code=404, detail="Primary dataset not loaded.")
        
    customer_record = df[df["SK_ID_CURR"] == sk_id]
    if customer_record.empty:
        raise HTTPException(status_code=404, detail=f"Customer with ID {sk_id} not found.")
        
    try:
        # Run prediction and SHAP explanation
        report = shap_service.explain(customer_record.drop(columns=["TARGET"], errors="ignore"))
        
        # Format profile dictionary to return to frontend (convert numpy/pandas types to standard native types)
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
                
        return RiskReportResponse(
            sk_id_curr=sk_id,
            default_probability=report["default_probability"],
            risk_category=report["risk_category"],
            financial_health_score=report["financial_health_score"],
            attributions=report["attributions"],
            profile=profile_dict
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

@router.post("/custom")
def predict_custom(req: CustomPredictRequest):
    """Evaluates a custom loan application from form features."""
    global shap_service
    if shap_service is None:
        shap_service = get_shap_service()
            
    try:
        custom_df = pd.DataFrame([req.features])
        # Ensure SK_ID_CURR is present
        if "SK_ID_CURR" not in custom_df:
            custom_df["SK_ID_CURR"] = 999999
            
        report = shap_service.explain(custom_df)
        return {
            "default_probability": report["default_probability"],
            "risk_category": report["risk_category"],
            "financial_health_score": report["financial_health_score"],
            "attributions": report["attributions"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
