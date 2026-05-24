"""Statistical hypothesis testing for A/B tests with error handling."""
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_claim_frequency_by_category(df, category_col, group_a, group_b):
    """
    Test if claim frequency differs between two groups.
    
    Args:
        df: DataFrame with insurance data
        category_col: Column name for categories
        group_a: First group value
        group_b: Second group value
    
    Returns:
        dict: Test results with p-value and statistics
    """
    try:
        # Validate inputs
        if category_col not in df.columns:
            raise ValueError(f"Column '{category_col}' not found in DataFrame")
        
        if group_a not in df[category_col].values:
            raise ValueError(f"Group '{group_a}' not found in {category_col}")
        
        if group_b not in df[category_col].values:
            raise ValueError(f"Group '{group_b}' not found in {category_col}")
        
        # Subset data
        subset = df[df[category_col].isin([group_a, group_b])].copy()
        
        if len(subset) == 0:
            raise ValueError("No data found for the specified groups")
        
        # Create contingency table
        contingency = pd.crosstab(subset[category_col], subset['HadClaim'])
        
        # Run chi-square test
        chi2, p, dof, expected = chi2_contingency(contingency)
        
        # Calculate rates
        rates = subset.groupby(category_col)['HadClaim'].mean()
        
        logger.info(f"Chi-square test completed for {category_col}: p={p:.4f}")
        
        return {
            'test': 'Chi-square',
            'statistic': chi2,
            'p_value': p,
            'reject_h0': p < 0.05,
            'group_a_rate': rates[group_a],
            'group_b_rate': rates[group_b],
            'difference': rates[group_a] - rates[group_b]
        }
    
    except Exception as e:
        logger.error(f"Error in test_claim_frequency_by_category: {e}")
        return {
            'test': 'Chi-square',
            'error': str(e),
            'p_value': 1.0,
            'reject_h0': False
        }


def test_margin_by_category(df, category_col, group_a, group_b):
    """
    Test if margin differs between two groups.
    """
    try:
        if category_col not in df.columns:
            raise ValueError(f"Column '{category_col}' not found")
        
        subset = df[df[category_col].isin([group_a, group_b])].copy()
        
        if len(subset) == 0:
            raise ValueError("No data found for specified groups")
        
        group_a_data = subset[subset[category_col] == group_a]['Margin'].dropna()
        group_b_data = subset[subset[category_col] == group_b]['Margin'].dropna()
        
        if len(group_a_data) == 0 or len(group_b_data) == 0:
            raise ValueError("Insufficient data for test")
        
        # Use Mann-Whitney U test (non-parametric, robust to outliers)
        stat, p = mannwhitneyu(group_a_data, group_b_data, alternative='two-sided')
        
        logger.info(f"Mann-Whitney test completed for {category_col}: p={p:.4f}")
        
        return {
            'test': "Mann-Whitney U",
            'statistic': stat,
            'p_value': p,
            'reject_h0': p < 0.05,
            'group_a_mean': group_a_data.mean(),
            'group_b_mean': group_b_data.mean(),
            'difference': group_a_data.mean() - group_b_data.mean()
        }
    
    except Exception as e:
        logger.error(f"Error in test_margin_by_category: {e}")
        return {
            'test': "Mann-Whitney U",
            'error': str(e),
            'p_value': 1.0,
            'reject_h0': False
        } 
