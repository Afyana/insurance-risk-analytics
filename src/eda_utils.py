"""EDA utilities for visualizations and summaries."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_data_summary(df):
    """Get comprehensive data summary."""
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing': df.isnull().sum().to_dict(),
        'missing_pct': (df.isnull().sum() / len(df) * 100).to_dict(),
        'numeric_stats': df.describe().to_dict(),
        'categorical_stats': {col: df[col].value_counts().head(10).to_dict() 
                              for col in df.select_dtypes(include=['object']).columns}
    }
    return summary


def plot_loss_ratio_by_category(df, category_col, figsize=(10, 6)):
    """Plot loss ratio by category."""
    fig, ax = plt.subplots(figsize=figsize)
    
    grouped = df.groupby(category_col).agg({
        'TotalClaims': 'sum',
        'TotalPremium': 'sum'
    }).reset_index()
    grouped['LossRatio'] = grouped['TotalClaims'] / grouped['TotalPremium']
    grouped = grouped.sort_values('LossRatio', ascending=False)
    
    colors = ['red' if x > 1 else 'green' for x in grouped['LossRatio']]
    bars = ax.bar(grouped[category_col].astype(str), grouped['LossRatio'], color=colors)
    ax.axhline(y=1, color='black', linestyle='--', label='Break-even (1.0)')
    ax.set_xlabel(category_col)
    ax.set_ylabel('Loss Ratio (Claims / Premium)')
    ax.set_title(f'Loss Ratio by {category_col}')
    ax.legend()
    
    for bar, val in zip(bars, grouped['LossRatio']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9, rotation=45)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


def plot_monthly_trends(df, metrics=['TotalPremium', 'TotalClaims'], figsize=(12, 6)):
    """Plot monthly trends for premiums and claims."""
    monthly = df.groupby('TransactionYearMonth').agg({
        'TotalPremium': 'sum',
        'TotalClaims': 'sum',
        'CustomerID': 'count'
    }).reset_index()
    monthly['LossRatio'] = monthly['TotalClaims'] / monthly['TotalPremium']
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Premiums and claims over time
    axes[0].plot(monthly['TransactionYearMonth'], monthly['TotalPremium'], 
                 marker='o', label='Total Premium', linewidth=2)
    axes[0].plot(monthly['TransactionYearMonth'], monthly['TotalClaims'], 
                 marker='s', label='Total Claims', linewidth=2)
    axes[0].set_xlabel('Month')
    axes[0].set_ylabel('Amount')
    axes[0].set_title('Monthly Premiums vs Claims')
    axes[0].legend()
    axes[0].tick_params(axis='x', rotation=45)
    
    # Loss ratio over time
    axes[1].plot(monthly['TransactionYearMonth'], monthly['LossRatio'], 
                 marker='o', color='red', linewidth=2)
    axes[1].axhline(y=1, color='black', linestyle='--', alpha=0.7)
    axes[1].set_xlabel('Month')
    axes[1].set_ylabel('Loss Ratio')
    axes[1].set_title('Monthly Loss Ratio')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig


def plot_claim_distribution_by_make(df, top_n=15, figsize=(12, 6)):
    """Plot top vehicle makes by average claim amount."""
    claim_by_make = df[df['TotalClaims'] > 0].groupby('AutoMake')['TotalClaims'].agg(['mean', 'count', 'sum'])
    claim_by_make = claim_by_make.sort_values('mean', ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.RdYlGn_r(claim_by_make['mean'] / claim_by_make['mean'].max())
    bars = ax.barh(claim_by_make.index, claim_by_make['mean'], color=colors)
    
    ax.set_xlabel('Average Claim Amount')
    ax.set_title(f'Top {top_n} Vehicle Makes by Average Claim Severity')
    ax.axvline(x=df[df['TotalClaims'] > 0]['TotalClaims'].median(), 
               color='black', linestyle='--', alpha=0.7, label='Median Claim')
    
    for bar, val in zip(bars, claim_by_make['mean']):
        ax.text(val + 100, bar.get_y() + bar.get_height()/2, 
                f'{val:,.0f}', va='center', fontsize=9)
    
    ax.legend()
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df, figsize=(12, 10)):
    """Plot correlation heatmap of numerical features."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=0.5, ax=ax)
    ax.set_title('Feature Correlation Matrix')
    plt.tight_layout()
    return fig