"""Statistical hypothesis testing for A/B tests."""
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ttest_ind, f_oneway
from scipy.stats import mannwhitneyu, kruskal


def test_claim_frequency_by_category(df, category_col, group_a, group_b):
    """
    Test if claim frequency differs between two groups.
    H0: No difference in claim frequency
    Returns: p-value and test statistic
    """
    subset = df[df[category_col].isin([group_a, group_b])]
    
    # Create contingency table
    contingency = pd.crosstab(subset[category_col], subset['HadClaim'])
    
    chi2, p, dof, expected = chi2_contingency(contingency)
    
    # Calculate rates
    rates = subset.groupby(category_col)['HadClaim'].mean()
    
    return {
        'test': 'Chi-square',
        'statistic': chi2,
        'p_value': p,
        'reject_h0': p < 0.05,
        'group_a_rate': rates[group_a],
        'group_b_rate': rates[group_b],
        'difference': rates[group_a] - rates[group_b]
    }


def test_claim_severity_by_category(df, category_col, group_a, group_b):
    """
    Test if claim severity differs between two groups.
    Only includes policies with claims > 0.
    """
    claims_df = df[df['TotalClaims'] > 0]
    subset = claims_df[claims_df[category_col].isin([group_a, group_b])]
    
    group_a_data = subset[subset[category_col] == group_a]['TotalClaims']
    group_b_data = subset[subset[category_col] == group_b]['TotalClaims']
    
    # Use Mann-Whitney U test (non-parametric, robust to outliers)
    stat, p = mannwhitneyu(group_a_data, group_b_data, alternative='two-sided')
    
    return {
        'test': 'Mann-Whitney U',
        'statistic': stat,
        'p_value': p,
        'reject_h0': p < 0.05,
        'group_a_mean': group_a_data.mean(),
        'group_b_mean': group_b_data.mean(),
        'difference': group_a_data.mean() - group_b_data.mean()
    }


def test_margin_by_category(df, category_col, group_a, group_b):
    """
    Test if margin differs between two groups.
    """
    subset = df[df[category_col].isin([group_a, group_b])]
    
    group_a_data = subset[subset[category_col] == group_a]['Margin']
    group_b_data = subset[subset[category_col] == group_b]['Margin']
    
    # Use t-test if data is roughly normal, otherwise Mann-Whitney
    stat, p = ttest_ind(group_a_data, group_b_data, equal_var=False)
    
    return {
        'test': "Welch's t-test",
        'statistic': stat,
        'p_value': p,
        'reject_h0': p < 0.05,
        'group_a_mean': group_a_data.mean(),
        'group_b_mean': group_b_data.mean(),
        'difference': group_a_data.mean() - group_b_data.mean()
    }


def test_multiple_categories(df, category_col, metric='LossRatio'):
    """
    Test if there are differences across multiple categories using ANOVA.
    """
    groups = []
    for category in df[category_col].unique():
        if metric == 'LossRatio':
            data = df[df[category_col] == category]['LossRatio'].dropna()
        elif metric == 'Margin':
            data = df[df[category_col] == category]['Margin']
        elif metric == 'TotalClaims':
            data = df[df['TotalClaims'] > 0]
            data = data[data[category_col] == category]['TotalClaims']
        else:
            data = df[df[category_col] == category][metric]
        
        if len(data) > 0:
            groups.append(data)
    
    # Kruskal-Wallis test (non-parametric)
    stat, p = kruskal(*groups)
    
    return {
        'test': 'Kruskal-Wallis',
        'statistic': stat,
        'p_value': p,
        'reject_h0': p < 0.05,
        'n_categories': len(groups)
    }


def run_all_hypothesis_tests(df):
    """Run all the required hypothesis tests."""
    results = []
    
    # Test 1: Provinces (using Gauteng vs Western Cape as example)
    # Note: Your data uses different province names - adjust as needed
    provinces = df['Province'].unique()
    if len(provinces) >= 2:
        results.append({
            'hypothesis': 'No risk differences across provinces',
            **test_multiple_categories(df, 'Province', 'LossRatio')
        })
    
    # Test 2: Zip codes (top 2 most frequent)
    top_zips = df['ZipCode'].value_counts().head(2).index.tolist()
    if len(top_zips) >= 2:
        results.append({
            'hypothesis': 'No risk differences between zip codes',
            **test_claim_frequency_by_category(df, 'ZipCode', top_zips[0], top_zips[1])
        })
    
    # Test 3: Margin difference between zip codes
    if len(top_zips) >= 2:
        results.append({
            'hypothesis': 'No significant margin difference between zip codes',
            **test_margin_by_category(df, 'ZipCode', top_zips[0], top_zips[1])
        })
    
    # Test 4: Gender risk difference
    if len(df['Gender'].unique()) >= 2:
        genders = df['Gender'].unique()[:2]
        results.append({
            'hypothesis': 'No significant risk difference between women and men',
            **test_claim_frequency_by_category(df, 'Gender', genders[0], genders[1])
        })
    
    return pd.DataFrame(results)