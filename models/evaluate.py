import argparse
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, classification_report, confusion_matrix
from sklearn.calibration import calibration_curve
from pathlib import Path

# Add project root to python path to avoid import errors
sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.ft_transformer import FTTransformer
from models.train import TabularDataset

def evaluate_model(test_path, model_path, device="cpu"):
    print("Loading test dataset...")
    test_dataset = TabularDataset(test_path)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    num_features = test_dataset.X.shape[1]
    
    print("Loading model weights...")
    model = FTTransformer(num_features=num_features, d_token=32, n_blocks=2, n_heads=4, d_ffn=64, dropout=0.1)
    if Path(model_path).exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Error: Model file not found at {model_path}")
        return
    model.to(device)
    model.eval()
    
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits)
            
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(y_batch.numpy())
            
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    all_preds = (all_probs > 0.5).astype(int)
    
    # Calculate Metrics
    roc_auc = roc_auc_score(all_targets, all_probs)
    
    precision_vals, recall_vals, _ = precision_recall_curve(all_targets, all_probs)
    pr_auc = auc(recall_vals, precision_vals)
    
    conf_matrix = confusion_matrix(all_targets, all_preds)
    class_report = classification_report(all_targets, all_preds)
    
    # Probability Calibration
    prob_true, prob_pred = calibration_curve(all_targets, all_probs, n_bins=10)
    
    print("\n================ EVALUATION METRICS ================")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)
    
    print("Calibration Curve details:")
    for i, (t, p) in enumerate(zip(prob_true, prob_pred)):
        print(f"  Bin {i+1}: True Prob={t:.4f}, Pred Prob={p:.4f}")
    print("====================================================")
    
    # Save metrics to reports file
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "evaluation_metrics.txt", "w") as f:
        f.write(f"ROC-AUC: {roc_auc:.4f}\n")
        f.write(f"PR-AUC: {pr_auc:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(conf_matrix) + "\n\n")
        f.write("Classification Report:\n")
        f.write(class_report + "\n")
        
    print(f"Metrics saved to {reports_dir / 'evaluation_metrics.txt'}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-data", default="data/processed/test.csv")
    ap.add_argument("--model-path", default="models/ft_transformer.pt")
    a = ap.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate_model(a.test_data, a.model_path, device=device)

if __name__ == "__main__":
    main()
