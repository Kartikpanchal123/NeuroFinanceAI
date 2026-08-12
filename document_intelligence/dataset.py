import os
import glob
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
from pathlib import Path

# Category labels mapping
CLASS_MAP = {
    "payslip": 0,
    "bank_statement": 1,
    "loan_statement": 2,
    "credit_card_statement": 3,
    "other": 4
}
INV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}

class DocumentDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.file_paths)
        
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Load image as RGB
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def generate_synthetic_document(label_name, file_path, width=224, height=224):
    """
    Generates a synthetic document image using PIL and saves a corresponding text file.
    """
    # Create white canvas
    img = Image.new('RGB', (width, height), color='#f3f4f6')
    draw = ImageDraw.Draw(img)
    
    # Text contents depending on class
    lines = []
    if label_name == "payslip":
        salary = random.randint(45, 180) * 1000
        employer = random.choice(["ABC Pvt Ltd", "DEF Corp", "Global Solutions", "Apex Financial"])
        emp_type = random.choice(["full_time", "part_time"])
        lines = [
            "OFFICIAL PAYSLIP",
            f"Employer: {employer}",
            f"Employment Type: {emp_type}",
            f"Monthly Income: {salary}",
            f"Net Pay: {salary} INR",
            f"Pay Date: 30-July-2026"
        ]
    elif label_name == "bank_statement":
        bal = random.randint(20, 500) * 1000
        dep = random.randint(10, 100) * 1000
        delinq = random.choice(["No", "No", "No", "Yes"])
        lines = [
            "TRUST BANK STATEMENT",
            "Account Holder: Client Profile",
            f"Average Balance: {bal}",
            f"Recent Deposits: {dep}",
            f"Delinquent Status: {delinq}"
        ]
    elif label_name == "loan_statement":
        outstanding = random.randint(100, 800) * 1000
        emi = random.randint(10, 50) * 1000
        lines = [
            "LOAN ACC SUMMARY",
            "Borrower: Client Profile",
            f"Outstanding Loan: {outstanding}",
            f"Monthly EMI Payment: {emi}",
            "Status: Active"
        ]
    elif label_name == "credit_card_statement":
        limit = random.randint(100, 500) * 1000
        bal = random.randint(10, 150) * 1000
        lines = [
            "APEX CARD INVOICE",
            f"Credit Limit: {limit}",
            f"Outstanding Balance: {bal}",
            "Minimum Due: 2500"
        ]
    else:  # other
        lines = [
            "UTILITY BILL / INVOICE",
            "Customer Name: John Doe",
            "Bill Period: July 2026",
            "Total Amount Due: 4500",
            "Status: PAID"
        ]
        
    text_content = "\n".join(lines)
    
    # Draw simple text lines on the image
    y_offset = 20
    for line in lines:
        draw.text((20, y_offset), line, fill="#0f172a")
        y_offset += 25
        
    # Draw a mock border and signature line to make it look like a document layout
    draw.rectangle([10, 10, width - 10, height - 10], outline=(180, 180, 180), width=2)
    draw.line([20, height - 30, width - 20, height - 30], fill=(180, 180, 180), width=1)
    
    # Save Image
    img.save(file_path, "JPEG")
    
    # Save corresponding .txt file next to it for OCR
    txt_path = Path(file_path).with_suffix('.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(text_content)

def setup_synthetic_dataset(data_dir="data/document_images", samples_per_class=35):
    """
    Creates directories and populates them with synthetic document images.
    """
    print(f"Dataset Setup: Generating {samples_per_class} synthetic images per class...")
    base_path = Path(data_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    for class_name in CLASS_MAP.keys():
        class_path = base_path / class_name
        class_path.mkdir(parents=True, exist_ok=True)
        
        for i in range(samples_per_class):
            img_filename = f"doc_{i}.jpg"
            img_path = class_path / img_filename
            generate_synthetic_document(class_name, img_path)
            
    print("Dataset Setup: Generation complete!")

def get_document_splits(data_dir="data/document_images", train_split=0.7, val_split=0.15, transform=None):
    """
    Splits all available document images into stratified sets and returns DocumentDatasets.
    """
    setup_synthetic_dataset(data_dir)
    
    all_files = []
    all_labels = []
    
    for class_name, label_idx in CLASS_MAP.items():
        files = glob.glob(os.path.join(data_dir, class_name, "*.jpg"))
        all_files.extend(files)
        all_labels.extend([label_idx] * len(files))
        
    # Pair and shuffle
    paired = list(zip(all_files, all_labels))
    random.seed(42)
    random.shuffle(paired)
    
    # Split sizes
    total = len(paired)
    train_size = int(total * train_split)
    val_size = int(total * val_split)
    
    train_pairs = paired[:train_size]
    val_pairs = paired[train_size:train_size + val_size]
    test_pairs = paired[train_size + val_size:]
    
    def unzip(pairs):
        if not pairs:
            return [], []
        files, labels = zip(*pairs)
        return list(files), list(labels)
        
    train_files, train_labels = unzip(train_pairs)
    val_files, val_labels = unzip(val_pairs)
    test_files, test_labels = unzip(test_pairs)
    
    # Datasets
    train_dataset = DocumentDataset(train_files, train_labels, transform=transform)
    val_dataset = DocumentDataset(val_files, val_labels, transform=transform)
    test_dataset = DocumentDataset(test_files, test_labels, transform=transform)
    
    print(f"Dataset Split Details:")
    print(f"  - Train Set Size: {len(train_dataset)}")
    print(f"  - Val Set Size:   {len(val_dataset)}")
    print(f"  - Test Set Size:  {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    # Test generation
    train, val, test = get_document_splits()
    print("DataLoader test completed successfully.")
