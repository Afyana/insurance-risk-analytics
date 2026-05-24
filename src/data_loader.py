import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "insurance_data.csv"


def load_data(file_path=None):
    """Load insurance dataset."""
    if file_path is None:
        file_path = DATA_PATH
    return pd.read_csv(file_path)


def clean_data(df):
    """Clean and preprocess the dataset."""
    df = df.copy()
    
    # Convert date
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
    df['TransactionMonth'] = df['TransactionDate'].dt.to_period('M')
    df['TransactionYearMonth'] = df['TransactionDate'].dt.strftime('%Y-%m')
    
    # Convert boolean columns
    df['Claimed'] = df['Claimed'].astype(bool)
    
    # Derive loss ratio and margin
    df['LossRatio'] = df['TotalClaims'] / df['TotalPremium'].replace(0, np.nan)
    df['Margin'] = df['TotalPremium'] - df['TotalClaims']
    
    # Flag if claim occurred
    df['HadClaim'] = (df['TotalClaims'] > 0).astype(int)
    
    # Claim severity (only for policies with claims)
    df['ClaimSeverity'] = df['TotalClaims'].where(df['HadClaim'] == 1, 0)
    
    return df


def get_feature_columns(df, exclude_targets=True):
    """Get feature columns for modeling."""
    exclude = ['CustomerID', 'TransactionDate', 'TransactionMonth', 'TransactionYearMonth',
               'LossRatio', 'Margin', 'HadClaim', 'ClaimSeverity']
    if exclude_targets:
        exclude.extend(['TotalClaims', 'TotalPremium', 'Claimed'])
    
    return [col for col in df.columns if col not in exclude]


def prepare_model_data(df, target='TotalClaims', test_size=0.2, random_state=42):
    """Prepare train/test split for modeling."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    
    df = clean_data(df)
    
    # For severity model (only claims > 0)
    if target == 'TotalClaims':
        df_model = df[df['TotalClaims'] > 0].copy()
    else:
        df_model = df.copy()
    
    # Get features and target
    feature_cols = get_feature_columns(df_model)
    X = df_model[feature_cols].copy()
    y = df_model[target]
    
    # Encode categorical variables
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test, feature_cols
