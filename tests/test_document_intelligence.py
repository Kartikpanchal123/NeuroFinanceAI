import os
import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.document_cnn import DocumentCNN
from document_intelligence.ocr import extract_information, validate_fields

def test_document_cnn_forward_shape():
    """Verifies that the DocumentCNN accepts 224x224 RGB tensors and outputs 5 logits."""
    model = DocumentCNN(num_classes=5)
    dummy_input = torch.randn(2, 3, 224, 224)
    logits = model(dummy_input)
    assert logits.shape == (2, 5)

def test_ocr_information_extraction_payslip():
    """Verifies that the OCR parser extracts monthly income, employer, and employment type from payslips."""
    sample_text = (
        "NEUROFINANCE CONFIDENTIAL PAYSLIP\n"
        "Employer: ABC Corp Services\n"
        "Net Pay: 85000\n"
        "Employment Type: full_time\n"
    )
    extracted = extract_information(sample_text, "PAYSLIP")
    assert extracted["monthly_income"] == 85000.0
    assert extracted["employer"] == "ABC Corp Services"
    assert extracted["employment_type"] == "full_time"

def test_ocr_information_extraction_bank():
    """Verifies that the OCR parser extracts average balances from bank statements."""
    sample_text = (
        "GLOBAL TRUST BANK ACCOUNT STATEMENT\n"
        "Average Balance: 125000\n"
        "Recent Deposits: 30000\n"
        "Delinquent Status: Yes\n"
    )
    extracted = extract_information(sample_text, "BANK_STATEMENT")
    assert extracted["average_balance"] == 125000.0
    assert extracted["recent_deposits"] == 30000.0
    assert extracted["delinquent_status"] == "yes"

def test_ocr_validation_payslip_warnings():
    """Verifies that the validation layer alerts on invalid or missing payslip fields."""
    # Test valid fields
    valid_fields = {"monthly_income": 75000, "employer": "ABC Pvt Ltd"}
    warnings = validate_fields(valid_fields, "PAYSLIP")
    assert len(warnings) == 0
    
    # Test missing fields
    invalid_fields = {"monthly_income": None, "employer": ""}
    warnings = validate_fields(invalid_fields, "PAYSLIP")
    assert len(warnings) > 0
    assert "Could not extract Monthly Income" in warnings[0]
