import re
from pathlib import Path

def extract_text(image_path):
    """
    Simulates OCR. It reads a corresponding text file (with the same name but .txt extension)
    next to the image if it exists. Otherwise, it generates mock text based on the directory name.
    """
    img_path = Path(image_path)
    txt_path = img_path.with_suffix('.txt')
    
    # 1. Try reading the associated text file
    if txt_path.exists():
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception:
            pass
            
    # 2. Mock generation fallback based on file path content
    path_str = str(img_path).lower()
    
    if "payslip" in path_str:
        return (
            "NEUROFINANCE OFFICIAL PAYSLIP\n"
            "Employer: ABC Technologies Pvt Ltd\n"
            "Employee Name: John Doe\n"
            "Employment Type: full_time\n"
            "Monthly Income: 75000\n"
            "Net Pay: 75000 INR\n"
            "Pay Date: 30-July-2026\n"
        )
    elif "bank_statement" in path_str:
        return (
            "GLOBAL TRUST BANK - STATEMENT OF ACCOUNT\n"
            "Account Holder: John Doe\n"
            "Average Balance: 120000\n"
            "Recent Deposits: 45000\n"
            "Delinquent Status: No\n"
            "Overdraft Limits: 0\n"
        )
    elif "loan_statement" in path_str:
        return (
            "CREST LOAN ACCOUNT SUMMARY\n"
            "Borrower ID: John Doe\n"
            "Outstanding Loan: 250000\n"
            "Monthly EMI Payment: 12500\n"
            "Status: Active\n"
        )
    elif "credit_card_statement" in path_str:
        return (
            "APEX CREDIT CARD MONTHLY INVOICE\n"
            "Credit Limit: 150000\n"
            "Outstanding Balance: 45000\n"
            "Minimum Due: 2500\n"
        )
        
    return "UNKNOWN DOCUMENT TYPE\nNo OCR text could be parsed.\n"

def extract_information(text, doc_type):
    """
    Extracts structured fields from raw OCR text using regular expressions.
    """
    text_lower = text.lower()
    fields = {}
    
    if doc_type == "PAYSLIP":
        # Extract Monthly Income
        income_match = re.search(r"(monthly income|net pay|salary):\s*(\d+)", text_lower)
        fields["monthly_income"] = float(income_match.group(2)) if income_match else None
        
        # Extract Employer
        employer_match = re.search(r"(employer|company):\s*([^\n\r]+)", text, re.IGNORECASE)
        fields["employer"] = employer_match.group(2).strip() if employer_match else None
        
        # Extract Employment Type
        emp_type_match = re.search(r"(employment type|type):\s*(\w+)", text_lower)
        fields["employment_type"] = emp_type_match.group(2).strip() if emp_type_match else "full_time"
        
    elif doc_type == "BANK_STATEMENT":
        # Extract Avg Balance
        bal_match = re.search(r"(average balance|avg balance):\s*(\d+)", text_lower)
        if not bal_match:
            bal_match = re.search(r"(?<!opening )balance:\s*(\d+)", text_lower)
        fields["average_balance"] = float(bal_match.group(2)) if bal_match else None
        
        # Extract Recent Deposits
        dep_match = re.search(r"(recent deposits|deposits):\s*(\d+)", text_lower)
        fields["recent_deposits"] = float(dep_match.group(2)) if dep_match else None
        
        # Extract Delinquent Status
        del_match = re.search(r"delinquent status:\s*(\w+)", text_lower)
        fields["delinquent_status"] = del_match.group(1).strip() if del_match else "no"
        
    elif doc_type == "LOAN_STATEMENT":
        # Extract Loan Balance
        loan_match = re.search(r"(outstanding loan|balance):\s*(\d+)", text_lower)
        fields["loan_balance"] = float(loan_match.group(2)) if loan_match else None
        
        # Extract EMI Paid
        emi_match = re.search(r"(monthly emi payment|emi):\s*(\d+)", text_lower)
        fields["emi_payment"] = float(emi_match.group(2)) if emi_match else None
        
    elif doc_type == "CREDIT_CARD_STATEMENT":
        # Extract Card Limit
        limit_match = re.search(r"(credit limit|limit):\s*(\d+)", text_lower)
        fields["credit_limit"] = float(limit_match.group(2)) if limit_match else None
        
        # Extract Balance
        bal_match = re.search(r"(outstanding balance|balance):\s*(\d+)", text_lower)
        fields["credit_balance"] = float(bal_match.group(2)) if bal_match else None
        
    return fields

def validate_fields(fields, doc_type):
    """
    Validates extracted values and returns a list of warnings (if any).
    """
    warnings = []
    
    if doc_type == "PAYSLIP":
        income = fields.get("monthly_income")
        if income is None:
            warnings.append("Could not extract Monthly Income from Payslip.")
        elif income <= 0:
            warnings.append("Extracted Monthly Income is invalid (must be positive).")
            
        employer = fields.get("employer")
        if not employer:
            warnings.append("Could not extract Employer name from Payslip.")
            
    elif doc_type == "BANK_STATEMENT":
        balance = fields.get("average_balance")
        if balance is None:
            warnings.append("Could not extract Average Balance from Bank Statement.")
        elif balance < 0:
            warnings.append("Extracted Average Balance is negative.")
            
        delinquent = fields.get("delinquent_status", "no")
        if delinquent in ["yes", "true", "1"]:
            warnings.append("Warning: Bank Statement indicates active loan delinquency status!")
            
    return warnings

if __name__ == "__main__":
    # Test OCR parsing
    sample_text = (
        "EMPLOYER: DEF Corporation\n"
        "Net Pay: 85000\n"
        "Type: part_time\n"
    )
    res = extract_information(sample_text, "PAYSLIP")
    warns = validate_fields(res, "PAYSLIP")
    print("OCR Extract Test:")
    print(f"Extracted: {res}")
    print(f"Warnings:  {warns}")
