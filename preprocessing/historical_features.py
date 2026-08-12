import pandas as pd
import numpy as np
from pathlib import Path

def aggregate_historical_features(data_dir="data/raw"):
    data_path = Path(data_dir)
    print("Aggregating historical features...")
    
    # 1. Bureau features
    bureau_path = data_path / "bureau.csv"
    if bureau_path.exists():
        print("Processing bureau.csv...")
        bureau = pd.read_csv(bureau_path)
        
        # previous_loan_count
        prev_loan_count = bureau.groupby("SK_ID_CURR").size().rename("previous_loan_count")
        
        # active_loan_count
        active_loans = bureau[bureau["CREDIT_ACTIVE"] == "Active"]
        active_loan_count = active_loans.groupby("SK_ID_CURR").size().rename("active_loan_count")
        
        bureau_feats = pd.concat([prev_loan_count, active_loan_count], axis=1).fillna(0)
    else:
        print("Warning: bureau.csv not found!")
        bureau_feats = pd.DataFrame(columns=["previous_loan_count", "active_loan_count"])
        bureau_feats.index.name = "SK_ID_CURR"
        
    # 2. Previous Applications features
    prev_app_path = data_path / "previous_application.csv"
    if prev_app_path.exists():
        print("Processing previous_application.csv...")
        prev_app = pd.read_csv(prev_app_path)
        
        # previous_application_count
        prev_app_count = prev_app.groupby("SK_ID_CURR").size().rename("previous_application_count")
        
        # previous_approval_rate
        approved = prev_app[prev_app["NAME_CONTRACT_STATUS"] == "Approved"]
        approved_count = approved.groupby("SK_ID_CURR").size()
        prev_app_rate = (approved_count / prev_app_count).rename("previous_approval_rate").fillna(0)
        
        # previous_credit_amount (average credit amount)
        prev_credit_amt = prev_app.groupby("SK_ID_CURR")["AMT_CREDIT"].mean().rename("previous_credit_amount").fillna(0)
        
        prev_feats = pd.concat([prev_app_count, prev_app_rate, prev_credit_amt], axis=1).fillna(0)
    else:
        print("Warning: previous_application.csv not found!")
        prev_feats = pd.DataFrame(columns=["previous_application_count", "previous_approval_rate", "previous_credit_amount"])
        prev_feats.index.name = "SK_ID_CURR"

    # 3. Installments Payments features
    inst_path = data_path / "installments_payments.csv"
    if inst_path.exists():
        print("Processing installments_payments.csv...")
        inst = pd.read_csv(inst_path)
        
        # average_payment_delay (actual payment date - scheduled installment date)
        # Positive values mean late payment
        inst["delay"] = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]
        avg_delay = inst.groupby("SK_ID_CURR")["delay"].mean().rename("average_payment_delay").fillna(0)
        
        # late_payment_count (days late > 0 or payment amount < installment amount)
        inst["is_late"] = (inst["delay"] > 0) | (inst["AMT_PAYMENT"] < inst["AMT_INSTALMENT"])
        late_count = inst.groupby("SK_ID_CURR")["is_late"].sum().rename("late_payment_count").fillna(0)
        
        inst_feats = pd.concat([avg_delay, late_count], axis=1).fillna(0)
    else:
        print("Warning: installments_payments.csv not found!")
        inst_feats = pd.DataFrame(columns=["average_payment_delay", "late_payment_count"])
        inst_feats.index.name = "SK_ID_CURR"

    # 4. Credit Card Balance features
    cc_path = data_path / "credit_card_balance.csv"
    if cc_path.exists():
        print("Processing credit_card_balance.csv...")
        cc = pd.read_csv(cc_path)
        
        # credit_utilization = average of (AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL)
        # Avoid division by zero
        limit = cc["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
        cc["utilization"] = (cc["AMT_BALANCE"] / limit).clip(lower=0, upper=2.0)
        cc_util = cc.groupby("SK_ID_CURR")["utilization"].mean().rename("credit_utilization").fillna(0)
        
        cc_feats = pd.DataFrame(cc_util)
    else:
        print("Warning: credit_card_balance.csv not found!")
        cc_feats = pd.DataFrame(columns=["credit_utilization"])
        cc_feats.index.name = "SK_ID_CURR"

    # Merge all features
    dfs = [bureau_feats, prev_feats, inst_feats, cc_feats]
    hist_feats = dfs[0]
    for df in dfs[1:]:
        hist_feats = hist_feats.join(df, how="outer")
        
    hist_feats = hist_feats.fillna(0)
    print(f"Historical features aggregated successfully! Shape: {hist_feats.shape}")
    return hist_feats

if __name__ == "__main__":
    df_hist = aggregate_historical_features()
    print(df_hist.head())
