import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

class NeuroBotRouter:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if self.api_key and self.api_key != "your_key_here":
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel("gemini-1.5-flash")
            print("NeuroBot Router: Gemini configured successfully!")
        else:
            self.model = None
            print("NeuroBot Router Warning: GEMINI_API_KEY is missing. Using local rule-based intent router.")

    def route_intent(self, query):
        """Routes the user's query to 'RISK', 'RAG', 'FINANCE', 'DOCUMENT', or 'GENERAL'."""
        query_lower = query.lower()
        
        # 1. Check Gemini-based classification if available
        if self.model is not None:
            prompt = (
                f"You are the NeuroBot Intent Router. Categorize the user's query into exactly one of these labels:\n"
                f"- RISK: If the user is asking to assess loan/default risk, explain risk factors, or predict credit probability.\n"
                f"- RAG: If the user is asking about banking policies, RBI regulations, required documents, or general credit rules.\n"
                f"- FINANCE: If the user is asking to calculate loan EMIs, interest rates, monthly payments, or loan affordability.\n"
                f"- DOCUMENT: If the user is asking to analyze an uploaded financial document (like a payslip or bank statement), extract fields, or run OCR validation risk checks.\n"
                f"- GENERAL: If the query is a greeting, general conversation, or general financial questions not covered by specific guidelines.\n\n"
                f"Output ONLY the word: RISK, RAG, FINANCE, DOCUMENT, or GENERAL.\n\n"
                f"Query: \"{query}\"\n"
                f"Label:"
            )
            try:
                response = self.model.generate_content(prompt)
                label = response.text.strip().upper()
                if label in ["RISK", "RAG", "FINANCE", "DOCUMENT", "GENERAL"]:
                    return label
            except Exception as e:
                print(f"NeuroBot Router: Gemini routing failed, using rule-based fallback: {e}")
                
        # 2. Rule-based fallback classification
        # Document detection
        if any(kw in query_lower for kw in ["document", "payslip", "bank statement", "upload", "ocr", "extract"]):
            return "DOCUMENT"
        # Finance detection
        if any(kw in query_lower for kw in ["calculate", "emi", "interest", "payment", "repay", "rate", "%"]):
            return "FINANCE"
        # Risk detection
        if any(kw in query_lower for kw in ["risk", "default", "assess", "score", "probability", "repay risk", "credit status"]):
            return "RISK"
        # Conversational greetings/general help checks
        if any(kw in query_lower for kw in ["hi", "hello", "hey", "who are you", "help", "greet", "how are you", "what is"]):
            return "GENERAL"
        # Default to RAG
        return "RAG"

if __name__ == "__main__":
    router = NeuroBotRouter()
    queries = [
        "Assess my loan risk.",
        "Calculate EMI for 8 lakh for 5 years at 9%.",
        "What documents are generally required for home loans?",
        "Why is my risk high?"
    ]
    print("\nIntent Routing Test:")
    for q in queries:
        print(f"Query: \"{q}\" -> Route: {router.route_intent(q)}")
