import os
import sys
from pathlib import Path
import glob

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from document_intelligence.predict import DocumentIntelligencePredictor
from backend.routes.prediction import get_raw_data, shap_service

class DocumentAgent:
    def __init__(self, model_path="models/document_cnn.pt"):
        self.predictor = DocumentIntelligencePredictor(model_path=model_path)
        
    def analyze_document_influence(self, image_path, customer_id):
        """
        Runs CNN+OCR on the uploaded document, merges the fields, runs the credit risk model,
        and returns a structured text report.
        """
        img_path = Path(image_path)
        if not img_path.exists():
            return {
                "success": False,
                "report": "Error: Uploaded document file could not be found."
            }
            
        # 1. Run inference
        try:
            doc_res = self.predictor.predict(str(img_path))
        except Exception as e:
            return {
                "success": False,
                "report": f"Error executing document inference: {e}"
            }
            
        doc_type = doc_res["document_type"]
        confidence = doc_res["confidence"]
        extracted = doc_res["extracted_fields"]
        warnings = doc_res["validation_warnings"]
        
        # 2. Get customer profile
        df = get_raw_data()
        if df.empty:
            return {
                "success": False,
                "report": "Error: Main customer dataset is not loaded."
            }
            
        customer_record = df[df["SK_ID_CURR"] == customer_id].copy()
        if customer_record.empty:
            return {
                "success": False,
                "report": f"Error: Customer ID {customer_id} not found."
            }
            
        # Get baseline risk
        from backend.routes.prediction import get_shap_service
        active_shap_service = get_shap_service()
        
        baseline_prob = 0.0
        baseline_health = 0.0
        if active_shap_service is not None:
            try:
                base_rep = active_shap_service.explain(customer_record.drop(columns=["TARGET"], errors="ignore"))
                baseline_prob = base_rep["default_probability"]
                baseline_health = base_rep["financial_health_score"]
            except Exception:
                pass
                
        # 3. Integrate fields
        integration_text = ""
        if doc_type == "PAYSLIP" and "monthly_income" in extracted and extracted["monthly_income"]:
            monthly_income = extracted["monthly_income"]
            annual_income = monthly_income * 12
            customer_record.loc[:, "AMT_INCOME_TOTAL"] = annual_income
            integration_text = f"Updated Borrower Net Annual Income to **Rs. {annual_income:,.2f}** (based on extracted monthly salary of Rs. {monthly_income:,.2f})."
        elif doc_type == "BANK_STATEMENT" and "average_balance" in extracted and extracted["average_balance"]:
            integration_text = f"Verified Bank Statement Average Balance of **Rs. {extracted['average_balance']:,.2f}**."
        else:
            integration_text = f"Analyzed {doc_type} document layout. No direct numeric overrides mapped to risk model features."
            
        # 4. Predict new risk
        new_prob = baseline_prob
        new_health = baseline_health
        new_category = "Unknown"
        factors_text = ""
        
        if active_shap_service is not None:
            try:
                new_rep = active_shap_service.explain(customer_record.drop(columns=["TARGET"], errors="ignore"))
                new_prob = new_rep["default_probability"]
                new_health = new_rep["financial_health_score"]
                new_category = new_rep["risk_category"]
                
                # Format attributions
                factors_text = "\n".join([f"- **{f['feature']}**: attribution {f['value']:.4f}" for f in new_rep["attributions"]["top_risk_factors"][:3]])
            except Exception as e:
                return {
                    "success": False,
                    "report": f"Error calculating simulated risk scores: {e}"
                }
        else:
            return {
                "success": False,
                "report": "Error: Risk scoring model is not initialized yet."
            }
            
        # 5. Compile warning text
        warning_bullet = "\n".join([f"- **Warning**: {w}" for w in warnings]) if warnings else "- *No validation warnings triggered. Verification checks passed.*"
        
        # 6. Format report
        report = (
            f"### Document Agent Intelligence Report\n\n"
            f"#### 1. CNN Classification & Verification\n"
            f"* **Identified Document**: **{doc_type}**\n"
            f"* **Classification Confidence**: {confidence:.2%}\n"
            f"* **Extracted Fields**: {extracted}\n\n"
            f"#### 2. Verification Alerts & Warnings\n"
            f"{warning_bullet}\n\n"
            f"#### 3. Financial Profile Integration\n"
            f"{integration_text}\n\n"
            f"#### 4. Simulated Credit Risk Impact\n"
            f"* **Baseline Default Probability**: {baseline_prob:.2%}\n"
            f"* **Simulated Default Probability**: **{new_prob:.2%}**\n"
            f"* **Financial Health Score**: {new_health}/100\n"
            f"* **Revised Risk Category**: **{new_category} Risk**\n\n"
            f"#### 5. Simulated Attributions\n"
            f"{factors_text}\n"
        )
        
        return {
            "success": True,
            "report": report,
            "data": {
                "document_type": doc_type,
                "confidence": confidence,
                "extracted_fields": extracted,
                "default_probability": new_prob,
                "risk_category": new_category,
                "financial_health_score": new_health
            }
        }
