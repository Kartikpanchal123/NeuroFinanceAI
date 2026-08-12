import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from preprocessing.historical_features import aggregate_historical_features

def add_financial_features(df):
    df = df.copy()
    
    # Age and employment duration in years
    df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365.25).clip(lower=0)
    
    # Handle DAYS_EMPLOYED anomaly (365243 represents missing/unemployed)
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
    days_employed_clean = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    df["EMPLOYMENT_YEARS"] = (-days_employed_clean / 365.25).clip(lower=0)
    
    # Financial ratios
    income = df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    df["CREDIT_TO_INCOME"] = df["AMT_CREDIT"] / income
    df["ANNUITY_TO_INCOME"] = df["AMT_ANNUITY"] / income
    df["GOODS_TO_INCOME"] = df["AMT_GOODS_PRICE"] / income
    
    # Income per family member
    family = df["CNT_FAM_MEMBERS"].replace(0, np.nan)
    df["INCOME_PER_FAMILY_MEMBER"] = df["AMT_INCOME_TOTAL"] / family
    
    # External source scores average
    ext = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df]
    if ext: 
        df["EXT_SOURCE_MEAN"] = df[ext].mean(axis=1)
        
    # Count missing values per row
    df["MISSING_VALUE_COUNT"] = df.isna().sum(axis=1)
    
    return df.replace([np.inf, -np.inf], np.nan)

class NeuroFinancePreprocessor:
    def __init__(self):
        self.num_imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.cat_imputer = SimpleImputer(strategy="most_frequent")
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.num_cols = []
        self.cat_cols = []
        self.feature_cols = []
        self.cat_encoded_cols = []
        self.all_processed_cols = []

    def fit(self, X_train):
        # We assume X_train has SK_ID_CURR as a column and it should NOT be scaled/encoded
        X = X_train.drop(columns=["SK_ID_CURR"])
        
        self.num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Fit numerical pipeline
        if self.num_cols:
            self.num_imputer.fit(X[self.num_cols])
            imputed_num = self.num_imputer.transform(X[self.num_cols])
            self.scaler.fit(imputed_num)
            
        # Fit categorical pipeline
        if self.cat_cols:
            self.cat_imputer.fit(X[self.cat_cols])
            imputed_cat = self.cat_imputer.transform(X[self.cat_cols])
            self.encoder.fit(imputed_cat)
            encoded_cat_names = self.encoder.get_feature_names_out(self.cat_cols).tolist()
            self.cat_encoded_cols = encoded_cat_names
        else:
            self.cat_encoded_cols = []
            
        self.all_processed_cols = self.num_cols + self.cat_encoded_cols

    def transform(self, df):
        df_out = pd.DataFrame(index=df.index)
        df_out["SK_ID_CURR"] = df["SK_ID_CURR"]
        
        X = df.drop(columns=["SK_ID_CURR"])
        
        # Transform numerical features
        if self.num_cols:
            num_data = X[self.num_cols]
            imputed_num = self.num_imputer.transform(num_data)
            scaled_num = self.scaler.transform(imputed_num)
            df_num = pd.DataFrame(scaled_num, columns=self.num_cols, index=df.index)
            df_out = pd.concat([df_out, df_num], axis=1)
            
        # Transform categorical features
        if self.cat_cols:
            cat_data = X[self.cat_cols]
            imputed_cat = self.cat_imputer.transform(cat_data)
            encoded_cat = self.encoder.transform(imputed_cat)
            df_cat = pd.DataFrame(encoded_cat, columns=self.cat_encoded_cols, index=df.index)
            df_out = pd.concat([df_out, df_cat], axis=1)
            
        return df_out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/application_train.csv")
    ap.add_argument("--output", default="data/processed")
    ap.add_argument("--include-historical", action="store_true", default=True)
    a = ap.parse_args()
    
    print(f"Loading raw applications from {a.input}...")
    df = pd.read_csv(a.input).drop_duplicates()
    
    # Separate label
    y = df.pop("TARGET").astype(int)
    
    # Financial features
    print("Engineering base financial features...")
    X = add_financial_features(df)
    
    # Optionally merge historical features
    if a.include_historical:
        hist_df = aggregate_historical_features(Path(a.input).parent)
        print("Merging historical features...")
        X = X.merge(hist_df, on="SK_ID_CURR", how="left")
        # For customers with no history, fill aggregated count/delays with 0
        fill_cols = [c for c in hist_df.columns if c in X]
        X[fill_cols] = X[fill_cols].fillna(0)
    
    print("Performing stratified train/validation/test split (80/10/10)...")
    # Stratified split to preserve class ratio
    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    
    # Fit preprocessor on training data only
    print("Fitting preprocessor pipeline on training data...")
    preprocessor = NeuroFinancePreprocessor()
    preprocessor.fit(X_train_raw)
    
    # Transform all splits
    print("Transforming datasets...")
    X_train = preprocessor.transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)
    
    # Write processed data
    out_dir = Path(a.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    train_df = X_train.assign(TARGET=y_train.values)
    val_df = X_val.assign(TARGET=y_val.values)
    test_df = X_test.assign(TARGET=y_test.values)
    
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "validation.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    
    # Save the fitted preprocessor pipeline to models/
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    with open(models_dir / "preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)
        
    print(f"Data preprocessing complete! Preprocessor saved to {models_dir / 'preprocessor.pkl'}")
    print(f"Train Shape: {train_df.shape} | Val Shape: {val_df.shape} | Test Shape: {test_df.shape}")
    print(f"Default rate: {y_train.mean():.4f}")

if __name__ == "__main__":
    main()
