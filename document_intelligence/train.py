import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import argparse

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.document_cnn import DocumentCNN
from document_intelligence.dataset import get_document_splits
from document_intelligence.preprocessing import get_train_transforms, get_eval_transforms

def train_cnn(data_dir="data/document_images", model_dir="models", epochs=3, batch_size=8, lr=1e-3, device="cpu"):
    print(f"CNN Training: Loading dataset splits on device: {device}...")
    
    # Load splits
    train_dataset, val_dataset, _ = get_document_splits(
        data_dir=data_dir,
        transform=get_train_transforms()
    )
    # Validation uses eval transforms
    val_dataset.transform = get_eval_transforms()
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize Model
    print("CNN Training: Initializing DocumentCNN baseline...")
    model = DocumentCNN(num_classes=5)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(logits, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
        train_loss /= len(train_dataset)
        train_acc = train_correct / train_total
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(logits, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
        val_loss /= len(val_dataset)
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            Path(model_dir).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), Path(model_dir) / "document_cnn.pt")
            print("  -> Saved best model checkpoint to models/document_cnn.pt")
            
    print("CNN Training: Finished!")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_cnn(epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, device=device)

if __name__ == "__main__":
    main()
