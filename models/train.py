import argparse
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Add project root to python path to avoid import errors
sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.ft_transformer import FTTransformer

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class TabularDataset(Dataset):
    def __init__(self, csv_path, subset=None):
        df = pd.read_csv(csv_path)
        if subset is not None and subset > 0:
            df = df.head(subset)
        # Drop client ID and target columns
        self.y = torch.tensor(df["TARGET"].values, dtype=torch.float32)
        X_df = df.drop(columns=["SK_ID_CURR", "TARGET"])
        self.X = torch.tensor(X_df.values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_model(train_path, val_path, model_dir="models", epochs=5, batch_size=256, lr=1e-4, device="cpu", subset=None):
    print(f"Loading datasets on device: {device}...")
    train_dataset = TabularDataset(train_path, subset=subset)
    val_subset = (subset // 5) if (subset is not None and subset > 0) else None
    val_dataset = TabularDataset(val_path, subset=val_subset)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    num_features = train_dataset.X.shape[1]
    print(f"Number of features: {num_features}")
    
    # Calculate pos_weight for BCEWithLogitsLoss to handle class imbalance
    # pos_weight = negative_count / positive_count
    neg_count = (train_dataset.y == 0).sum().item()
    pos_count = (train_dataset.y == 1).sum().item()
    pos_weight = neg_count / pos_count
    print(f"Class imbalance weights - Negatives: {neg_count}, Positives: {pos_count}, PosWeight: {pos_weight:.4f}")
    
    model = FTTransformer(num_features=num_features, d_token=32, n_blocks=2, n_heads=4, d_ffn=64, dropout=0.1)
    model.to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_val_loss = float("inf")
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            
        train_loss /= len(train_dataset)
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
                
        val_loss /= len(val_dataset)
        val_acc = correct / total
        
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            Path(model_dir).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), Path(model_dir) / "ft_transformer.pt")
            print("Model saved to models/ft_transformer.pt")
            
    print("Training finished!")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-data", default="data/processed/train.csv")
    ap.add_argument("--val-data", default="data/processed/validation.csv")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--subset", type=int, default=0)
    a = ap.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    subset = a.subset if a.subset > 0 else None
    train_model(a.train_data, a.val_data, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, device=device, subset=subset)

if __name__ == "__main__":
    main()
