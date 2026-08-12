# NeuroFinance AI

### Financial Decision Intelligence Platform
**Transformer-based Credit Risk + RAG + Agentic AI**

NeuroFinance AI predicts loan-default risk from the Home Credit Default Risk dataset using a Transformer designed for tabular financial data. It adds SHAP explainability, a RAG financial knowledge base, and an EVA-like agentic assistant called NeuroBot.

## Core Features
- Home Credit preprocessing and financial feature engineering
- TabTransformer / FT-Transformer-style credit-risk model
- Probability of Default and risk categories
- SHAP Explainable AI
- Financial Health Score
- RAG over trusted financial/RBI documents
- Agentic NeuroBot with intent routing
- EMI and affordability tools
- Interactive dashboard
- Docker/GCP deployment

## Architecture
```text
Home Credit -> Preprocessing -> Transformer -> Risk Prediction
                                      |
                                     SHAP
                                      |
                                  NeuroBot
                              /      |       \
                         Risk Agent RAG Agent Finance Agent
                              \      |       /
                               Response
                                  |
                               Dashboard
                                  |
                                  GCP
```

## Dataset
**Home Credit Default Risk — Kaggle**

Raw files are not committed because of size. Place them locally under `data/raw/`.

Expected files:
- application_train.csv
- bureau.csv
- bureau_balance.csv
- previous_application.csv
- installments_payments.csv
- POS_CASH_balance.csv
- credit_card_balance.csv

## Setup
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env`. Never commit `.env`.

## Roadmap
1. EDA/preprocessing
2. Historical feature engineering
3. Transformer
4. SHAP
5. RAG
6. NeuroBot agents
7. Financial tools
8. Dashboard
9. Docker
10. GCP

## Disclaimer
Academic decision-support prototype; not an official credit-bureau score or autonomous lending system.
