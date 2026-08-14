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
    import hashlib
    # Generate a stable numerical seed based on the file name to keep results consistent for the same file
    fn_hash = int(hashlib.md5(img_path.name.encode('utf-8')).hexdigest(), 16)
    
    # Check if a specific monthly salary or balance is embedded in the filename (e.g. Payslip_85000.jpg)
    embedded_num = None
    num_matches = re.findall(r'\b\d{4,6}\b', img_path.name)
    if num_matches:
        embedded_num = int(num_matches[0])
    
    if "payslip" in path_str or "salary" in path_str or "income" in path_str:
        salary = embedded_num if embedded_num else 45000 + (fn_hash % 81) * 1000  # 45k to 125k INR
        employers = ["ABC Technologies Pvt Ltd", "TCS Ltd", "Infosys Limited", "Cognizant Technology", "Wipro Limited"]
        employer = employers[fn_hash % len(employers)]
        return (
            f"NEUROFINANCE OFFICIAL PAYSLIP\n"
            f"Employer: {employer}\n"
            f"Employee Name: Kartik Panchal\n"
            f"Employment Type: full_time\n"
            f"Monthly Income: {salary}\n"
            f"Net Pay: {salary} INR\n"
            f"Pay Date: 30-July-2026\n"
        )
    elif "bank" in path_str or "statement" in path_str:
        balance = embedded_num if (embedded_num and embedded_num > 5000) else 50000 + (fn_hash % 251) * 1000  # 50k to 300k INR
        deposits = 10000 + (fn_hash % 71) * 1000  # 10k to 80k INR
        delinquent = "Yes" if (fn_hash % 12 == 0) else "No"  # ~8% chance of delinquency alert
        banks = ["GLOBAL TRUST BANK", "HDFC BANK", "ICICI BANK", "SBI BANK", "AXIS BANK"]
        bank = banks[fn_hash % len(banks)]
        return (
            f"{bank} - STATEMENT OF ACCOUNT\n"
            f"Account Holder: Kartik Panchal\n"
            f"Average Balance: {balance}\n"
            f"Recent Deposits: {deposits}\n"
            f"Delinquent Status: {delinquent}\n"
            f"Overdraft Limits: 0\n"
        )
    elif "loan" in path_str:
        loan_amount = embedded_num if embedded_num else 100000 + (fn_hash % 41) * 10000  # 100k to 500k INR
        emi = int(loan_amount * 0.05)
        return (
            f"CREST LOAN ACCOUNT SUMMARY\n"
            f"Borrower Name: Kartik Panchal\n"
            f"Outstanding Loan: {loan_amount}\n"
            f"Monthly EMI Payment: {emi}\n"
            f"Status: Active\n"
        )
    elif "card" in path_str or "credit" in path_str:
        limit = embedded_num if embedded_num else 100000 + (fn_hash % 16) * 10000  # 100k to 250k INR
        balance = int(limit * (0.1 + (fn_hash % 6) * 0.1))
        return (
            f"APEX CREDIT CARD MONTHLY INVOICE\n"
            f"Credit Limit: {limit}\n"
            f"Outstanding Balance: {balance}\n"
            f"Minimum Due: {int(balance * 0.05)}\n"
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
