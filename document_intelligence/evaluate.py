import os
import sys
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.document_cnn import DocumentCNN
from document_intelligence.dataset import get_document_splits, INV_CLASS_MAP
from document_intelligence.preprocessing import get_eval_transforms

def evaluate_cnn(data_dir="data/document_images", model_path="models/document_cnn.pt", device="cpu"):
    print("CNN Evaluation: Loading test dataset...")
    
    # Load splits
    _, _, test_dataset = get_document_splits(
        data_dir=data_dir,
        transform=get_eval_transforms()
    )
    
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Initialize Model
    print("CNN Evaluation: Loading model weights...")
    model = DocumentCNN(num_classes=5)
    if Path(model_path).exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded weights from {model_path}")
    else:
        print(f"Error: Model file not found at {model_path}. Train the model first.")
        return
        
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            _, preds = torch.max(logits, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate Metrics in pure NumPy/Python to avoid scikit-learn MemoryErrors
    accuracy = (all_preds == all_labels).mean() if len(all_labels) > 0 else 0.0
    
    # Confusion Matrix
    conf_mat = np.zeros((5, 5), dtype=np.int32)
    for true_lbl, pred_lbl in zip(all_labels, all_preds):
        conf_mat[true_lbl, pred_lbl] += 1
        
    # Class-wise precision, recall, f1
    precision_list = []
    recall_list = []
    f1_list = []
    support_list = []
    
    class_report_lines = []
    class_report_lines.append(f"{'Class':<25} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    class_report_lines.append("-" * 65)
    
    for c in range(5):
        tp = np.sum((all_labels == c) & (all_preds == c))
        fp = np.sum((all_labels != c) & (all_preds == c))
        fn = np.sum((all_labels == c) & (all_preds != c))
        support = np.sum(all_labels == c)
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        
        precision_list.append(prec)
        recall_list.append(rec)
        f1_list.append(f1)
        support_list.append(support)
        
        c_name = INV_CLASS_MAP[c].upper()
        class_report_lines.append(f"{c_name:<25} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f} {support:<10d}")
        
    total_support = sum(support_list)
    if total_support > 0:
        weighted_prec = sum(p * s for p, s in zip(precision_list, support_list)) / total_support
        weighted_recall = sum(r * s for r, s in zip(recall_list, support_list)) / total_support
        weighted_f1 = sum(f * s for f, s in zip(f1_list, support_list)) / total_support
    else:
        weighted_prec, weighted_recall, weighted_f1 = 0.0, 0.0, 0.0
        
    class_report_lines.append("-" * 65)
    class_report_lines.append(f"{'Weighted Average':<25} {weighted_prec:<10.4f} {weighted_recall:<10.4f} {weighted_f1:<10.4f} {total_support:<10d}")
    
    class_report = "\n".join(class_report_lines)
    
    print("\n================ CNN EVALUATION METRICS ================")
    print(f"Overall Accuracy:  {accuracy:.2%}")
    print(f"Weighted Precision: {weighted_prec:.4f}")
    print(f"Weighted Recall:    {weighted_recall:.4f}")
    print(f"Weighted F1-Score:  {weighted_f1:.4f}")
    print("\nConfusion Matrix:")
    print(conf_mat)
    print("\nClassification Report:")
    print(class_report)
    print("========================================================")
    
    # Save metrics
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / "document_intelligence_metrics.txt"
    
    with open(report_file, "w") as f:
        f.write("CNN Document Intelligence Evaluation Report\n")
        f.write("===========================================\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision: {weighted_prec:.4f}\n")
        f.write(f"Recall:    {weighted_recall:.4f}\n")
        f.write(f"F1-Score:  {weighted_f1:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(conf_mat) + "\n\n")
        f.write("Classification Report:\n")
        f.write(class_report + "\n")
        
    print(f"Metrics saved to {report_file}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate_cnn(device=device)
