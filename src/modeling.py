"""Predictive modeling for claim severity and probability."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import shap


def train_claim_probability_model(X_train, X_test, y_train, y_test):
    """
    Train models to predict probability of a claim.
    y should be binary (1 if claim occurred, 0 otherwise)
    """
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    results = {}
    predictions = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        predictions[name] = y_pred_proba
    
    return models, results, predictions


def train_claim_severity_model(X_train, X_test, y_train, y_test):
    """
    Train models to predict claim severity (claims > 0 only).
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    predictions = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        results[name] = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
        predictions[name] = y_pred
    
    return models, results, predictions


def calculate_risk_premium(prob_claim, pred_severity, expense_loading=0.15, profit_margin=0.10):
    """
    Calculate risk-based premium:
    Premium = P(claim) × Predicted Severity + Expense Loading + Profit Margin
    """
    base_premium = prob_claim * pred_severity
    total_premium = base_premium * (1 + expense_loading + profit_margin)
    return total_premium


def get_feature_importance_shap(model, X_train, X_test, feature_names):
    """
    Calculate SHAP values for model interpretability.
    """
    if isinstance(model, (RandomForestRegressor, RandomForestClassifier)):
        explainer = shap.TreeExplainer(model)
    elif isinstance(model, xgb.XGBRegressor) or isinstance(model, xgb.XGBClassifier):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.LinearExplainer(model, X_train)
    
    shap_values = explainer.shap_values(X_test)
    
    # For classification, shap_values might be a list
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Take positive class
    
    # Get mean absolute SHAP values for feature importance
    importance = pd.DataFrame({
        'feature': feature_names,
        'shap_importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('shap_importance', ascending=False)
    
    return shap_values, importance


def compare_models_table(prob_results, severity_results):
    """Create comparison table for all models."""
    comparison = pd.DataFrame()
    
    # Probability models
    for name, metrics in prob_results.items():
        comparison.loc[name, 'Model Type'] = 'Claim Probability'
        comparison.loc[name, 'Accuracy'] = f"{metrics['accuracy']:.3f}"
        comparison.loc[name, 'Precision'] = f"{metrics['precision']:.3f}"
        comparison.loc[name, 'Recall'] = f"{metrics['recall']:.3f}"
        comparison.loc[name, 'F1'] = f"{metrics['f1']:.3f}"
        comparison.loc[name, 'ROC-AUC'] = f"{metrics['roc_auc']:.3f}"
    
    # Severity models
    for name, metrics in severity_results.items():
        comparison.loc[name, 'Model Type'] = 'Claim Severity'
        comparison.loc[name, 'RMSE'] = f"{metrics['rmse']:,.0f}"
        comparison.loc[name, 'R²'] = f"{metrics['r2']:.3f}"
    
    return comparison