import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.document_cnn import DocumentCNN
from document_intelligence.dataset import INV_CLASS_MAP
from document_intelligence.preprocessing import get_eval_transforms
from document_intelligence.ocr import extract_text, extract_information, validate_fields

class DocumentIntelligencePredictor:
    def __init__(self, model_path="models/document_cnn.pt"):
        self.device = "cpu"  # Keep CPU-based for fast API inferences
        self.transform = get_eval_transforms()
        
        # Load CNN Model
        print("Document Intelligence Predictor: Initializing DocumentCNN classifier...")
        self.model = DocumentCNN(num_classes=5)
        if Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Document Intelligence Predictor: Loaded CNN weights from {model_path}")
        else:
            print(f"Document Intelligence Predictor Warning: Weights file {model_path} not found. Classifier will output untrained results.")
            
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_path, top_k=3):
        """
        Runs document intelligence pipeline:
        Image -> CNN Classification -> OCR -> Extraction -> Validation -> Unified JSON.
        """
        img_path = Path(image_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Document image not found at {image_path}")
            
        # 1. Run CNN Classification
        # Load image
        image = Image.open(img_path).convert('RGB')
        transformed = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(transformed)
            probs = F.softmax(logits, dim=1).squeeze(0)
            
        # Get sorted predictions
        sorted_indices = torch.argsort(probs, descending=True)
        
        predictions = []
        for idx in sorted_indices[:top_k]:
            class_idx = idx.item()
            prob = probs[class_idx].item()
            predictions.append({
                "class": INV_CLASS_MAP[class_idx].upper(),
                "confidence": round(prob, 4)
            })
            
        primary_class = INV_CLASS_MAP[sorted_indices[0].item()].upper()
        primary_confidence = round(probs[sorted_indices[0].item()].item(), 4)
        
        # 2. Run OCR & Extraction
        raw_text = extract_text(image_path)
        extracted_fields = extract_information(raw_text, primary_class)
        validation_warnings = validate_fields(extracted_fields, primary_class)
        
        return {
            "document_type": primary_class,
            "confidence": primary_confidence,
            "predictions": predictions,
            "raw_text": raw_text,
            "extracted_fields": extracted_fields,
            "validation_warnings": validation_warnings
        }

if __name__ == "__main__":
    # Quick test run on the synthetic folder if trained
    predictor = DocumentIntelligencePredictor()
    sample_img = "data/document_images/payslip/doc_0.jpg"
    if Path(sample_img).exists():
        res = predictor.predict(sample_img)
        print("\nUnified Inference Result:")
        print(f"Document Type: {res['document_type']} (Confidence: {res['confidence']:.2%})")
        print(f"Extracted:     {res['extracted_fields']}")
        print(f"Warnings:      {res['validation_warnings']}")
    else:
        print("Run setup_synthetic_dataset or training first to test prediction.")
