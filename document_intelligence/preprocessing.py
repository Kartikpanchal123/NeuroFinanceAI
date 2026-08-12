import numpy as np
import torch

def preprocess_image(image, is_training=False):
    """
    Processes a PIL Image: resizes it, converts to RGB, and normalizes it.
    Returns a PyTorch tensor of shape (3, 224, 224).
    No torchvision dependency.
    """
    # Resize
    image = image.resize((224, 224))
    
    # Convert to numpy array and scale to [0, 1]
    arr = np.array(image, dtype=np.float32) / 255.0
    
    # Transpose from (H, W, C) to (C, H, W)
    arr = arr.transpose((2, 0, 1))
    
    # Standard ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    arr = (arr - mean) / std
    
    return torch.from_numpy(arr)

class LocalTrainTransform:
    def __call__(self, img):
        return preprocess_image(img, is_training=True)

class LocalEvalTransform:
    def __call__(self, img):
        return preprocess_image(img, is_training=False)

def get_train_transforms():
    return LocalTrainTransform()

def get_eval_transforms():
    return LocalEvalTransform()
