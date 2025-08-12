import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import csv
import re

def detect_csv_dialect(filepath):
    with open(filepath, 'r', newline='') as f:
        sample = f.read(1024)  # read first 1KB
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample)
        has_header = sniffer.has_header(sample)
    return dialect.delimiter, has_header


def load_data(file_path):
    try:
        delimiter, has_header = detect_csv_dialect(file_path)
        header = 0 if has_header else None
        df = pd.read_csv(file_path, delimiter=delimiter, header=header)
    except Exception as e:
        print(f"Warning: Could not detect dialect, loading with default settings. Error: {e}")
        df = pd.read_csv(file_path)
    return df


def drop_columns_with_many_nans(df, threshold=0.3):
    missing_frac = df.isna().mean()
    cols_to_keep = missing_frac[missing_frac <= threshold].index
    return df[cols_to_keep]

def detect_and_convert_dates(df, threshold=0.8):
    date_pattern = re.compile(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}")  # basic YYYY-MM-DD or similar
    for col in df.columns:
        if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
            if df[col].astype(str).str.match(date_pattern).mean() >= threshold:
                converted = pd.to_datetime(df[col], errors='coerce')
                success_ratio = converted.notna().mean()
                if success_ratio >= threshold:
                    df[col] = converted
    return df

def clean_data(df):
    # Normalize column names
    df.columns = [re.sub(r'\W+', '_', col.strip().lower()) for col in df.columns]
    
    
    # Strip whitespace from string values
    for col in df.select_dtypes(include=['object']):
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace(
        {"nan": np.nan, "NaN": np.nan, "": np.nan, "None": np.nan}) # Replace NaNs with np.nan
    

    # Replace inf values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Drop columns with too many NaNs
    df = drop_columns_with_many_nans(df)

    # Drop duplicates
    df = df.drop_duplicates()

    # Detect and convert dates
    df = detect_and_convert_dates(df)

    # Separate numeric, categorical
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    # Fill missing numeric with median
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Fill missing categorical with mode or 'Unknown'
    for col in categorical_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val[0] if not mode_val.empty else 'Unknown')

    # Convert categorical columns to category dtype
    for col in categorical_cols:
        df[col] = df[col].astype('category')

    # Reset index
    df = df.reset_index(drop=True)

    return df



def encode_dataframe(df, target_col, cardinality_threshold=10):
    
    df = df.copy()
    
    encoders = {"onehot": {}, "label": {}, "target": None}
    
    # Encode target with LabelEncoder
    target_encoder = LabelEncoder()
    df[target_col] = target_encoder.fit_transform(df[target_col])
    encoders["target"] = target_encoder
    
    # Separate features and target
    features = df.drop(columns=[target_col])
    
    # Process each column
    for col in features.columns:
        if df[col].dtype == "object" or str(df[col].dtype) == "category":
            unique_vals = df[col].nunique()
            
            if unique_vals <= cardinality_threshold and unique_vals > 2:
                # One-Hot Encode
                ohe = OneHotEncoder(sparse_output=False, drop='first')  
                ohe_df = pd.DataFrame(
                    ohe.fit_transform(df[[col]]),
                    columns=[f"{col}_{cat}" for cat in ohe.categories_[0][1:]],
                    index=df.index
                )
                df = pd.concat([df.drop(columns=[col]), ohe_df], axis=1)
                encoders["onehot"][col] = ohe
                
            else:
                # Label Encode
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                encoders["label"][col] = le
    
    return df, encoders
