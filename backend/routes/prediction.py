from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from pathlib import Path
from explainability.shap_service import NeuroFinanceSHAPService

router = APIRouter(prefix="/api/prediction", tags=["prediction"])

# Initialize SHAP service and load dataset for lookups
shap_service = None
raw_data = None

try:
    # Initialize the SHAP service
    if Path("models/ft_transformer.pt").exists() and Path("models/preprocessor.pkl").exists():
        shap_service = NeuroFinanceSHAPService()
    else:
        print("Prediction API Router Warning: Model weights or preprocessor not found. SHAP service will start in lazy load mode.")
except Exception as e:
    print(f"Prediction API Router Error: Could not initialize SHAP service: {e}")

def get_raw_data():
    global raw_data
    if raw_data is None:
        raw_path = Path("data/raw/application_train.csv")
        if raw_path.exists():
            print("Prediction API Router: Loading application_train.csv for lookups...")
            raw_data = pd.read_csv(raw_path)
            print(f"Prediction API Router: Loaded {len(raw_data)} customer profiles.")
        else:
            print("Prediction API Router Warning: application_train.csv not found.")
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
    """Looks up a customer profile by ID, predicts default probability, and explains risk factors using SHAP."""
    global shap_service
    # Lazy load SHAP service if model trained in the background
    if shap_service is None:
        if Path("models/ft_transformer.pt").exists() and Path("models/preprocessor.pkl").exists():
            try:
                shap_service = NeuroFinanceSHAPService()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to initialize explainability service: {e}")
        else:
            raise HTTPException(status_code=503, detail="Model training is not complete yet.")

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
        if Path("models/ft_transformer.pt").exists() and Path("models/preprocessor.pkl").exists():
            shap_service = NeuroFinanceSHAPService()
        else:
            raise HTTPException(status_code=503, detail="Model is not trained yet.")
            
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
