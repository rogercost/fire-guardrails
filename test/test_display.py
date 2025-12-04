"""Tests for display.py helper functions."""

import datetime
import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from display import _fmt_currency, _resolve_analysis_horizon


class TestFmtCurrency:
    """Tests for _fmt_currency function."""

    def test_positive_value(self):
        assert _fmt_currency(1000.0) == "$1,000"
        assert _fmt_currency(1234567.89) == "$1,234,568"

    def test_negative_value(self):
        assert _fmt_currency(-1000.0) == "-$1,000"
        assert _fmt_currency(-500.50) == "-$500"

    def test_zero(self):
        assert _fmt_currency(0.0) == "$0"

    def test_none_returns_na(self):
        assert _fmt_currency(None) == "N/A"

    def test_nan_returns_na(self):
        assert _fmt_currency(float('nan')) == "N/A"

    def test_inf_returns_na(self):
        assert _fmt_currency(float('inf')) == "N/A"
        assert _fmt_currency(float('-inf')) == "N/A"

    def test_escape_for_markdown(self):
        assert _fmt_currency(1000.0, escape_for_markdown=True) == "\\$1,000"
        assert _fmt_currency(-500.0, escape_for_markdown=True) == "-\\$500"

    def test_integer_input(self):
        assert _fmt_currency(1000) == "$1,000"

    def test_large_values(self):
        assert _fmt_currency(1_000_000_000.0) == "$1,000,000,000"


class TestResolveAnalysisHorizon:
    """Tests for _resolve_analysis_horizon function."""

    @pytest.fixture
    def mock_shiller_df(self):
        # Create a mock DataFrame with Date column
        dates = pd.date_range(start='1871-01-01', end='2024-10-01', freq='MS')
        return pd.DataFrame({'Date': dates})

    def test_simulation_mode_uses_start_date(self, mock_shiller_df):
        params = {
            'duration_months': 360,
            'start_date': pd.Timestamp('2000-01-01'),
        }
        num_months, analysis_end = _resolve_analysis_horizon(
            mock_shiller_df, params, is_guidance=False, context="test"
        )
        assert num_months == 360
        assert analysis_end == pd.Timestamp('2000-01-01')

    def test_simulation_mode_raises_on_invalid_duration(self, mock_shiller_df):
        params = {
            'duration_months': 0,
            'start_date': pd.Timestamp('2000-01-01'),
        }
        with pytest.raises(ValueError) as exc_info:
            _resolve_analysis_horizon(mock_shiller_df, params, is_guidance=False, context="test")
        assert "Invalid retirement duration to compute test" in str(exc_info.value)

    def test_simulation_mode_negative_duration(self, mock_shiller_df):
        params = {
            'duration_months': -10,
            'start_date': pd.Timestamp('2000-01-01'),
        }
        with pytest.raises(ValueError):
            _resolve_analysis_horizon(mock_shiller_df, params, is_guidance=False, context="test")

    def test_guidance_mode_uses_latest_shiller_date(self, mock_shiller_df):
        params = {
            'duration_months': 360,
            'start_date': pd.Timestamp('2000-01-01'),
        }
        num_months, analysis_end = _resolve_analysis_horizon(
            mock_shiller_df, params, is_guidance=True, context="test"
        )
        assert num_months == 360
        # Should use the latest date from shiller_df (2024-10-01) or today, whichever is earlier
        expected_max = pd.Timestamp('2024-10-01')
        today = pd.Timestamp(datetime.date.today())
        expected = expected_max if expected_max <= today else today
        assert analysis_end == expected

    def test_guidance_mode_defaults_to_360_on_zero_duration(self, mock_shiller_df):
        params = {
            'duration_months': 0,
            'start_date': pd.Timestamp('2000-01-01'),
        }
        num_months, _ = _resolve_analysis_horizon(
            mock_shiller_df, params, is_guidance=True, context="test"
        )
        assert num_months == 360

    def test_guidance_mode_defaults_to_360_on_negative_duration(self, mock_shiller_df):
        params = {
            'duration_months': -10,
            'start_date': pd.Timestamp('2000-01-01'),
        }
        num_months, _ = _resolve_analysis_horizon(
            mock_shiller_df, params, is_guidance=True, context="test"
        )
        assert num_months == 360

    def test_context_appears_in_error_message(self, mock_shiller_df):
        params = {
            'duration_months': 0,
            'start_date': pd.Timestamp('2000-01-01'),
        }
        with pytest.raises(ValueError) as exc_info:
            _resolve_analysis_horizon(
                mock_shiller_df, params, is_guidance=False, context="initial spending rate"
            )
        assert "initial spending rate" in str(exc_info.value)
