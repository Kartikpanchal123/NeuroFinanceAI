import torch
import torch.nn as nn
import torch.nn.functional as F

class DocumentCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(DocumentCNN, self).__init__()
        
        # Conv block 1: Input 3x224x224 -> Output 32x112x112
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv block 2: Input 32x112x112 -> Output 64x56x56
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv block 3: Input 64x56x56 -> Output 128x28x28
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Adaptive pooling to force spatial size to 7x7 regardless of input
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Fully Connected Head
        self.fc = nn.Linear(128 * 7 * 7, 512)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Conv layers
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv3(x))
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Dense layers
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        logits = self.classifier(x)
        
        # Note: BCEWithLogitsLoss or CrossEntropyLoss in PyTorch takes logits directly.
        # We output raw logits. For inference, we apply softmax.
        return logits

if __name__ == "__main__":
    model = DocumentCNN()
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print("CNN Architecture Test:")
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
