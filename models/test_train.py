import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.ft_transformer import FTTransformer
from models.train import TabularDataset

def main():
    print("Step 1: Loading test dataset...")
    train_path = "data/processed/train.csv"
    if not Path(train_path).exists():
        print(f"Error: train.csv not found at {train_path}")
        return
        
    # Let's load only the first 1000 rows to test memory/loader issues
    dataset = TabularDataset(train_path)
    print(f"Dataset loaded. Total samples: {len(dataset)}")
    
    # Take a small subset of 1000 samples to keep it fast
    indices = torch.arange(1000)
    subset_dataset = torch.utils.data.Subset(dataset, indices)
    
    loader = DataLoader(subset_dataset, batch_size=32, shuffle=True)
    num_features = dataset.X.shape[1]
    print(f"Step 2: Instantiating model with {num_features} features...")
    
    model = FTTransformer(num_features=num_features, d_token=32, n_blocks=2, n_heads=4, d_ffn=64, dropout=0.1)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    print("Step 3: Running a test training step...")
    model.train()
    for batch_idx, (X_batch, y_batch) in enumerate(loader):
        print(f"  Batch {batch_idx + 1}: X shape={X_batch.shape}, y shape={y_batch.shape}")
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        print(f"  Batch {batch_idx + 1}: Loss = {loss.item():.4f}")
        break  # Test only one batch
        
    print("Training test successful!")

if __name__ == "__main__":
    main()
