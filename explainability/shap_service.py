import os
import pickle
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.ft_transformer import FTTransformer
from preprocessing.preprocess import NeuroFinancePreprocessor

class NeuroFinanceSHAPService:
    def __init__(self, model_path="models/ft_transformer.pt", preprocessor_path="models/preprocessor.pkl", train_path="data/processed/train.csv"):
        self.device = "cpu"  # Keep explainability on CPU for API reliability
        
        # Load Preprocessor
        print("SHAP Service: Loading preprocessor...")
        import preprocessing.preprocess
        sys.modules['__main__'].NeuroFinancePreprocessor = preprocessing.preprocess.NeuroFinancePreprocessor
        with open(preprocessor_path, "rb") as f:
            self.preprocessor = pickle.load(f)
            
        # Determine number of features
        self.num_features = len(self.preprocessor.all_processed_cols)
        print(f"SHAP Service: Expected features = {self.num_features}")
        
        # Load Model
        print("SHAP Service: Loading model...")
        self.model = FTTransformer(num_features=self.num_features, d_token=32, n_blocks=2, n_heads=4, d_ffn=64, dropout=0.1)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Prepare background data for SHAP (first 50 samples from train)
        print("SHAP Service: Preparing background dataset...")
        try:
            train_df = pd.read_csv(train_path, nrows=50)
            X_bg = train_df.drop(columns=["SK_ID_CURR", "TARGET"]).values
            self.background_tensor = torch.tensor(X_bg, dtype=torch.float32).to(self.device)
        except Exception as e:
            print(f"SHAP Service Warning: Could not load background dataset: {e}")
            self.background_tensor = torch.zeros((50, self.num_features), dtype=torch.float32)
            
        # Initialize SHAP explainer
        self.explainer = None
        try:
            import shap
            # shap.DeepExplainer requires a PyTorch model and background dataset
            self.explainer = shap.DeepExplainer(self.model, self.background_tensor)
            print("SHAP Service: shap.DeepExplainer initialized successfully!")
        except Exception as e:
            print(f"SHAP Service Warning: Failed to initialize shap.DeepExplainer: {e}. Fallback to gradient-based attribution is ready.")

    def explain(self, raw_customer_df, top_k=5):
        """
        Takes a raw customer DataFrame (1 row), preprocesses it, predicts probability of default,
        and computes feature importance attributions.
        """
        # 1. Preprocess the record
        from preprocessing.preprocess import add_financial_features
        # Ensure we run base engineering
        eng_df = add_financial_features(raw_customer_df)
        
        # Ensure all columns required by the preprocessor are present
        for col in self.preprocessor.num_cols + self.preprocessor.cat_cols:
            if col not in eng_df:
                eng_df[col] = np.nan
                
        # Transform
        processed_df = self.preprocessor.transform(eng_df)
        X_eval = processed_df.drop(columns=["SK_ID_CURR"]).values
        x_tensor = torch.tensor(X_eval, dtype=torch.float32).to(self.device)
        
        # 2. Run prediction
        with torch.no_grad():
            logits = self.model(x_tensor)
            prob = torch.sigmoid(logits).item()
            
        # Calculate risk category and financial health score
        risk_category = "Low"
        if prob > 0.35:
            risk_category = "High"
        elif prob > 0.15:
            risk_category = "Medium"
            
        health_score = round(100.0 - (prob * 100.0), 2)
        
        # 3. Compute attributions (SHAP values or Fallback Input-Times-Gradient)
        attributions = None
        method_used = "DeepSHAP"
        
        if self.explainer is not None:
            try:
                # shap_values shape: [1, num_features]
                shap_values = self.explainer.shap_values(x_tensor)
                if isinstance(shap_values, list):
                    attributions = shap_values[0][0]
                else:
                    attributions = shap_values[0]
            except Exception as e:
                print(f"SHAP Service Warning: shap calculation failed, falling back to gradients: {e}")
                attributions = None
                
        # Fallback to Input-Times-Gradient
        if attributions is None:
            method_used = "Input-Times-Gradient"
            x_tensor.requires_grad = True
            logits_eval = self.model(x_tensor)
            self.model.zero_grad()
            logits_eval.backward()
            grads = x_tensor.grad.detach().cpu().numpy()[0]
            inputs = x_tensor.detach().cpu().numpy()[0]
            attributions = inputs * grads  # Input-Times-Gradient
            
        # 4. Map attributions to feature names
        feature_names = self.preprocessor.all_processed_cols
        attr_dict = dict(zip(feature_names, attributions))
        
        # Sort attributions
        sorted_attrs = sorted(attr_dict.items(), key=lambda item: item[1])
        
        # Top factors increasing risk (most positive attributions)
        top_risk_factors = [{"feature": f, "value": float(val)} for f, val in reversed(sorted_attrs) if val > 0][:top_k]
        # Top factors reducing risk (most negative attributions)
        top_saving_factors = [{"feature": f, "value": float(val)} for f, val in sorted_attrs if val < 0][:top_k]
        
        # Return structured explanation report
        return {
            "default_probability": round(prob, 4),
            "risk_category": risk_category,
            "financial_health_score": health_score,
            "method": method_used,
            "attributions": {
                "top_risk_factors": top_risk_factors,
                "top_saving_factors": top_saving_factors
            }
        }

if __name__ == "__main__":
    # Test service locally if train dataset and preprocessor exist
    if Path("models/ft_transformer.pt").exists() and Path("models/preprocessor.pkl").exists():
        service = NeuroFinanceSHAPService()
        raw_app = pd.read_csv("data/raw/application_train.csv", nrows=1).drop(columns=["TARGET"])
        report = service.explain(raw_app)
        print("\nSHAP Explanation Test:")
        print(f"Default Probability: {report['default_probability']:.4f}")
        print(f"Risk Category:       {report['risk_category']}")
        print(f"Health Score:        {report['financial_health_score']}")
        print("\nTop Factors Increasing Risk:")
        for factor in report['attributions']['top_risk_factors']:
            print(f"  - {factor['feature']}: {factor['value']:.4f}")
    else:
        print("Model and preprocessor must be trained first.")
