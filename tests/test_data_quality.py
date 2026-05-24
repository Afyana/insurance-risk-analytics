"""Data quality tests for insurance analytics."""
import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.data_loader import load_data, clean_data, validate_data


class TestDataQuality:
    """Test suite for data quality checks."""
    
    @classmethod
    def setup_class(cls):
        """Load data once for all tests."""
        cls.df_raw = load_data()
        cls.df_clean = clean_data(cls.df_raw)
    
    def test_no_empty_dataframe(self):
        """Test that data loads successfully."""
        assert self.df_raw is not None
        assert len(self.df_raw) > 0
    
    def test_required_columns_exist(self):
        """Test that all required columns are present."""
        required = ['TotalPremium', 'TotalClaims', 'Gender', 'Province', 'VehicleType']
        for col in required:
            assert col in self.df_clean.columns, f"Missing column: {col}"
    
    def test_no_negative_premiums(self):
        """Test that premiums are non-negative."""
        assert (self.df_clean['TotalPremium'] >= 0).all()
    
    def test_no_negative_claims(self):
        """Test that claims are non-negative."""
        assert (self.df_clean['TotalClaims'] >= 0).all()
    
    def test_loss_ratio_range(self):
        """Test that loss ratio is within reasonable range."""
        loss_ratio = self.df_clean['LossRatio'].dropna()
        assert (loss_ratio >= 0).all()
        assert (loss_ratio < 10).all()  # Sanity check
    
    def test_margin_calculation(self):
        """Test that margin = premium - claims."""
        margin_calc = self.df_clean['TotalPremium'] - self.df_clean['TotalClaims']
        pd.testing.assert_series_equal(
            margin_calc, self.df_clean['Margin'], 
            check_names=False
        )
    
    def test_gender_values(self):
        """Test that gender has only valid values."""
        valid_genders = ['Male', 'Female']
        assert self.df_clean['Gender'].isin(valid_genders).all()
    
    def test_claim_flag_consistency(self):
        """Test that HadClaim flag matches TotalClaims > 0."""
        expected_flag = (self.df_clean['TotalClaims'] > 0).astype(int)
        pd.testing.assert_series_equal(
            expected_flag, self.df_clean['HadClaim'],
            check_names=False
        )


class TestDataValidation:
    """Test data validation functions."""
    
    def test_validation_returns_dict(self):
        """Test that validate_data returns a dictionary."""
        df = load_data()
        df_clean = clean_data(df)
        results = validate_data(df_clean)
        assert isinstance(results, dict)
    
    def test_validation_has_expected_keys(self):
        """Test that validation results have expected keys."""
        df = load_data()
        df_clean = clean_data(df)
        results = validate_data(df_clean)
        expected_keys = ['has_missing', 'missing_count', 'negative_premium', 
                        'negative_claims', 'valid_loss_ratio']
        for key in expected_keys:
            assert key in results


if __name__ == "__main__":
    pytest.main([__file__, '-v']) 
