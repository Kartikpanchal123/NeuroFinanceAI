import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.ft_transformer import FTTransformer

def main():
    print("Testing FT-Transformer instantiation...")
    model = FTTransformer(num_features=261)
    print("Model created successfully!")
    
    print("Running forward pass with random input...")
    x = torch.randn(4, 261)
    logits = model(x)
    print("Forward pass successful!")
    print(f"Logits shape: {logits.shape}")
    print(f"Logits: {logits}")

if __name__ == "__main__":
    main()
