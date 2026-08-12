import re
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from pathlib import Path
import glob
import os

from agents.neurobot import NeuroBotRouter
from agents.document_agent import DocumentAgent
from backend.routes.prediction import predict_customer
from backend.routes.rag import run_query, QueryRequest
from backend.routes.tools import get_emi, EMIRequest
import google.generativeai as genai

# Initialize intent router and document agent placeholder
router = APIRouter(prefix="/api/neurobot", tags=["neurobot"])
bot_router = NeuroBotRouter()
doc_agent = None

class ChatRequest(BaseModel):
    query: str
    customer_id: Optional[int] = 100002  # Default sample customer

class ChatResponse(BaseModel):
    intent: str
    answer: str
    data: Optional[Dict[str, Any]] = None

def parse_finance_params_with_gemini(query: str, api_key: str) -> Optional[dict]:
    """Uses Gemini to parse principal, annual_rate, and months from user query."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = (
        f"Extract loan parameters from this query: \"{query}\"\n"
        f"Convert values to numerical formats: \n"
        f"- Principal in Rupees (e.g. 8 lakh or 8 lacs = 800000).\n"
        f"- Annual rate as a percentage float (e.g. 9% = 9.0).\n"
        f"- Tenure converted to total months (e.g. 5 years = 60).\n\n"
        f"Output format MUST be strict JSON: \n"
        f"{{\"principal\": float, \"annual_rate\": float, \"months\": int}}\n"
        f"If a parameter cannot be found, set it to 0. Output ONLY the JSON."
    )
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        # Find JSON boundaries
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            import json
            params = json.loads(match.group(0))
            return params
    except Exception as e:
        print(f"NeuroBot Chat: Gemini parameter parsing failed: {e}")
    return None

def parse_finance_params_regex(query: str) -> dict:
    """Fallback regex parser for EMI calculations."""
    query_lower = query.lower()
    
    # 1. Parse Principal
    principal = 0.0
    # Look for lakh
    lakh_match = re.search(r"(\d+\.?\d*)\s*(lakh|lacs|lac|l)", query_lower)
    if lakh_match:
        principal = float(lakh_match.group(1)) * 100000
    else:
        # Standard digits
        digits_match = re.findall(r"\b\d{4,9}\b", query_lower)
        if digits_match:
            principal = float(digits_match[0])
            
    # 2. Parse Rate
    rate = 0.0
    rate_match = re.search(r"(\d+\.?\d*)\s*(%|percent|interest|rate)", query_lower)
    if rate_match:
        rate = float(rate_match.group(1))
    else:
        # Look for floating numbers
        floats = re.findall(r"\b\d+\.\d+\b", query_lower)
        if floats:
            rate = float(floats[0])
        else:
            # Look for single digits like "at 9"
            at_match = re.search(r"at\s+(\d+)", query_lower)
            if at_match:
                rate = float(at_match.group(1))

    # 3. Parse tenure
    months = 0
    # Check years
    years_match = re.search(r"(\d+)\s*(year|years|yrs|yr|y)", query_lower)
    if years_match:
        months = int(years_match.group(1)) * 12
    else:
        # Check months directly
        months_match = re.search(r"(\d+)\s*(month|months|mths|mth|m)", query_lower)
        if months_match:
            months = int(months_match.group(1))
            
    return {
        "principal": principal if principal > 0 else 800000.0,
        "annual_rate": rate if rate > 0 else 9.0,
        "months": months if months > 0 else 60
    }

@router.post("/chat", response_model=ChatResponse)
def handle_chat(req: ChatRequest):
    # Route the query
    intent = bot_router.route_intent(req.query)
    
    if intent == "RISK":
        # Risk assessment for customer_id
        if not req.customer_id:
            return ChatResponse(
                intent=intent,
                answer="Please specify or select a valid Customer ID to perform a credit risk evaluation."
            )
        try:
            report = predict_customer(req.customer_id)
            
            # Format factors
            risk_factors = "\n".join([f"- **{f['feature']}**: attribution {f['value']:.4f}" for f in report.attributions["top_risk_factors"]])
            saving_factors = "\n".join([f"- **{f['feature']}**: attribution {f['value']:.4f}" for f in report.attributions["top_saving_factors"]])
            
            answer = (
                f"### Credit Risk Assessment Report — Customer ID {req.customer_id}\n\n"
                f"* **Default Probability**: {report.default_probability:.2%}\n"
                f"* **Risk Category**: {report.risk_category} Risk\n"
                f"* **Financial Health Score**: {report.financial_health_score}/100\n\n"
                f"#### Top Factors Increasing Risk:\n{risk_factors}\n\n"
                f"#### Top Factors Reducing Risk:\n{saving_factors}\n\n"
                f"This customer is categorized as **{report.risk_category} Risk** based on their default probability of **{report.default_probability:.2%}**. "
                f"You can view their full profile in the prediction dashboard."
            )
            return ChatResponse(intent=intent, answer=answer, data=report.dict())
        except Exception as e:
            # If model is not trained yet, return a helpful warning
            return ChatResponse(
                intent=intent,
                answer=f"Could not perform credit risk check: the model is currently training. Please wait a moment."
            )
            
    elif intent == "RAG":
        # Search the knowledge base
        try:
            res = run_query(QueryRequest(query=req.query))
            source_list = ", ".join(res.sources) if res.sources else "None"
            answer = (
                f"{res.answer}\n\n"
                f"*(Sources: {source_list})*"
            )
            return ChatResponse(intent=intent, answer=answer)
        except Exception as e:
            return ChatResponse(
                intent=intent,
                answer=f"RAG lookup failed: {e}. Standard KYC and lending rules apply."
            )
            
    elif intent == "FINANCE":
        # Financial / EMI calculation
        api_key = os.getenv("GEMINI_API_KEY")
        params = None
        
        # Try parsing with Gemini
        if api_key and api_key != "your_key_here":
            params = parse_finance_params_with_gemini(req.query, api_key)
            
        # Fallback to regex
        if not params or params.get("principal", 0) == 0:
            params = parse_finance_params_regex(req.query)
            
        try:
            emi_res = get_emi(EMIRequest(
                principal=params["principal"],
                annual_rate=params["annual_rate"],
                months=params["months"]
            ))
            
            answer = (
                f"### EMI Calculator Results\n\n"
                f"For a loan amount of **Rs. {params['principal']:,.2f}** at **{params['annual_rate']}%** annual interest for **{params['months']} months** ({params['months']/12:.1f} years):\n\n"
                f"* **Monthly EMI**: Rs. {emi_res.emi:,.2f}\n"
                f"* **Total Interest Payable**: Rs. {emi_res.total_interest:,.2f}\n"
                f"* **Total Payments (Principal + Interest)**: Rs. {emi_res.total_payment:,.2f}\n\n"
                f"Would you like to analyze if this monthly payment fits your income-to-annuity affordability ratio?"
            )
            return ChatResponse(intent=intent, answer=answer, data=emi_res.dict())
        except Exception as e:
            return ChatResponse(
                intent=intent,
                answer=f"Failed to calculate EMI: {e}. Make sure you specify the principal, interest rate, and tenure."
            )
            
    elif intent == "DOCUMENT":
        global doc_agent
        if doc_agent is None:
            try:
                doc_agent = DocumentAgent()
            except Exception as e:
                return ChatResponse(
                    intent=intent,
                    answer=f"Could not load Document Intelligence Agent: {e}"
                )
                
        # Look for the most recently uploaded image in data/temp_uploads/
        img_files = (
            glob.glob("data/temp_uploads/*.jpg") + 
            glob.glob("data/temp_uploads/*.png") + 
            glob.glob("data/temp_uploads/*.jpeg")
        )
        
        if not img_files:
            answer = (
                "### Upload a Document First 🤖\n"
                "I couldn't locate any uploaded document files in the session upload directory.\n\n"
                "Please follow these simple steps to perform an analysis:\n"
                "1. Click on the **'Document Intelligence'** tab in the dashboard.\n"
                "2. Upload a document image (like a payslip or bank statement).\n"
                "3. Select a borrower profile and click **'Analyze Financial Risk'**.\n"
                "4. Once uploaded, you can ask me: *'Analyze this document and tell me my financial risk'* and I will generate a complete credit influence report!"
            )
            return ChatResponse(intent=intent, answer=answer)
            
        newest_file = max(img_files, key=os.path.getmtime)
        try:
            cust_id = req.customer_id if req.customer_id else 100002
            res = doc_agent.analyze_document_influence(newest_file, cust_id)
            if res["success"]:
                return ChatResponse(intent=intent, answer=res["report"], data=res["data"])
            else:
                return ChatResponse(intent=intent, answer=f"Document analysis failed:\n{res['report']}")
        except Exception as e:
            return ChatResponse(intent=intent, answer=f"Document Agent error: {e}")
            
    return ChatResponse(intent="Unknown", answer="I'm sorry, I couldn't route your request. Please ask about loan risk, calculator details, or KYC document requirements.")
