"""Tests for utils.py core functions."""

import datetime
import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import utils


class TestParseShillerDate:
    """Tests for parse_shiller_date function."""

    def test_standard_two_digit_month(self):
        series = pd.Series(["1950.01", "1950.12"])
        result = utils.parse_shiller_date(series)
        assert result[0] == pd.Timestamp("1950-01-01")
        assert result[1] == pd.Timestamp("1950-12-01")

    def test_single_digit_october_special_case(self):
        # .1 means October (Excel dropped the zero)
        series = pd.Series(["1950.1"])
        result = utils.parse_shiller_date(series)
        assert result[0] == pd.Timestamp("1950-10-01")

    def test_numeric_input(self):
        series = pd.Series([1950.01, 1950.11])
        result = utils.parse_shiller_date(series)
        assert result[0] == pd.Timestamp("1950-01-01")
        assert result[1] == pd.Timestamp("1950-11-01")


class TestNeedsUpdate:
    """Tests for needs_update function."""

    def test_nonexistent_file_needs_update(self, tmp_path):
        fake_path = tmp_path / "nonexistent.txt"
        assert utils.needs_update(fake_path) is True

    def test_fresh_file_does_not_need_update(self, tmp_path):
        fresh_file = tmp_path / "fresh.txt"
        fresh_file.write_text("test")
        assert utils.needs_update(fresh_file, days=30) is False

    def test_custom_days_threshold(self, tmp_path):
        fresh_file = tmp_path / "fresh.txt"
        fresh_file.write_text("test")
        # With 0 days threshold, even fresh files need update
        assert utils.needs_update(fresh_file, days=0) is True


class TestCashflowScheduleForWindow:
    """Tests for cashflow_schedule_for_window function."""

    def test_empty_cashflows(self):
        result = utils.cashflow_schedule_for_window([], 12)
        assert len(result) == 12
        assert np.all(result == 0.0)

    def test_none_cashflows(self):
        result = utils.cashflow_schedule_for_window(None, 12)
        assert len(result) == 12
        assert np.all(result == 0.0)

    def test_single_cashflow_full_window(self):
        cashflows = [{"start_month": 0, "end_month": 11, "amount": 100.0}]
        result = utils.cashflow_schedule_for_window(cashflows, 12)
        assert len(result) == 12
        assert np.all(result == 100.0)

    def test_single_cashflow_partial_window(self):
        cashflows = [{"start_month": 0, "end_month": 5, "amount": 100.0}]
        result = utils.cashflow_schedule_for_window(cashflows, 12)
        assert np.all(result[:6] == 100.0)
        assert np.all(result[6:] == 0.0)

    def test_multiple_overlapping_cashflows(self):
        cashflows = [
            {"start_month": 0, "end_month": 11, "amount": 100.0},
            {"start_month": 6, "end_month": 11, "amount": 50.0},
        ]
        result = utils.cashflow_schedule_for_window(cashflows, 12)
        assert np.all(result[:6] == 100.0)
        assert np.all(result[6:] == 150.0)

    def test_cashflow_with_start_offset(self):
        cashflows = [{"start_month": 6, "end_month": 11, "amount": 100.0}]
        # Window starts at month 6, so cashflow should be at beginning of window
        result = utils.cashflow_schedule_for_window(cashflows, 6, start_offset=6)
        assert np.all(result == 100.0)

    def test_invalid_cashflow_end_before_start(self):
        cashflows = [{"start_month": 10, "end_month": 5, "amount": 100.0}]
        result = utils.cashflow_schedule_for_window(cashflows, 12)
        assert np.all(result == 0.0)

    def test_cashflow_outside_window(self):
        cashflows = [{"start_month": 20, "end_month": 25, "amount": 100.0}]
        result = utils.cashflow_schedule_for_window(cashflows, 12)
        assert np.all(result == 0.0)

    def test_zero_window_length(self):
        cashflows = [{"start_month": 0, "end_month": 5, "amount": 100.0}]
        result = utils.cashflow_schedule_for_window(cashflows, 0)
        assert len(result) == 0


class TestComputePortfolioReturns:
    """Tests for compute_portfolio_returns function."""

    def test_all_stocks(self):
        stock_prices = np.array([100.0, 110.0, 121.0])
        bond_prices = np.array([100.0, 102.0, 104.0])
        result = utils.compute_portfolio_returns(stock_prices, bond_prices, 1.0)
        # First return is always 1.0, then stock returns
        assert result[0] == 1.0
        assert np.isclose(result[1], 1.10)
        assert np.isclose(result[2], 1.10)

    def test_all_bonds(self):
        stock_prices = np.array([100.0, 110.0, 121.0])
        bond_prices = np.array([100.0, 102.0, 104.04])
        result = utils.compute_portfolio_returns(stock_prices, bond_prices, 0.0)
        assert result[0] == 1.0
        assert np.isclose(result[1], 1.02)
        assert np.isclose(result[2], 1.02)

    def test_balanced_portfolio(self):
        stock_prices = np.array([100.0, 110.0])  # 10% return
        bond_prices = np.array([100.0, 102.0])   # 2% return
        result = utils.compute_portfolio_returns(stock_prices, bond_prices, 0.5)
        # 50% * 1.10 + 50% * 1.02 = 1.06
        assert result[0] == 1.0
        assert np.isclose(result[1], 1.06)


class TestIsAdjustmentMonth:
    """Tests for is_adjustment_month function."""

    def test_monthly_always_true(self):
        for month in range(1, 13):
            ts = pd.Timestamp(f"2020-{month:02d}-01")
            assert utils.is_adjustment_month(ts, "Monthly") is True

    def test_quarterly(self):
        # Q1 starts: Jan, Apr, Jul, Oct
        quarterly_months = [1, 4, 7, 10]
        for month in range(1, 13):
            ts = pd.Timestamp(f"2020-{month:02d}-01")
            expected = month in quarterly_months
            assert utils.is_adjustment_month(ts, "Quarterly") is expected

    def test_biannually(self):
        biannual_months = [1, 7]
        for month in range(1, 13):
            ts = pd.Timestamp(f"2020-{month:02d}-01")
            expected = month in biannual_months
            assert utils.is_adjustment_month(ts, "Biannually") is expected

    def test_annually(self):
        for month in range(1, 13):
            ts = pd.Timestamp(f"2020-{month:02d}-01")
            expected = month == 1
            assert utils.is_adjustment_month(ts, "Annually") is expected

    def test_unknown_frequency_defaults_to_monthly(self):
        ts = pd.Timestamp("2020-06-01")
        assert utils.is_adjustment_month(ts, "Unknown") is True


class TestTestAllPeriods:
    """Tests for the numba-compiled test_all_periods function."""

    def test_constant_returns_no_withdrawal(self):
        # With no withdrawals and positive returns, should always succeed
        portfolio_returns = np.array([1.01] * 12)  # 1% monthly return
        monthly_cashflows = np.zeros(6, dtype=np.float64)
        result = utils.test_all_periods(
            portfolio_returns, 6, 100000.0, 0.0, monthly_cashflows
        )
        assert result == 1.0  # 100% success

    def test_high_withdrawal_fails(self):
        # Flat returns with withdrawals exceeding value should fail
        portfolio_returns = np.array([1.0] * 12)  # No growth
        monthly_cashflows = np.zeros(6, dtype=np.float64)
        # Withdraw more than initial value over 6 months
        result = utils.test_all_periods(
            portfolio_returns, 6, 100000.0, 20000.0, monthly_cashflows
        )
        assert result == 0.0  # 0% success

    def test_cashflows_offset_withdrawals(self):
        # Cashflows that fully offset withdrawals should succeed
        portfolio_returns = np.array([1.0] * 12)
        monthly_cashflows = np.array([5000.0] * 6, dtype=np.float64)
        result = utils.test_all_periods(
            portfolio_returns, 6, 100000.0, 5000.0, monthly_cashflows
        )
        assert result == 1.0  # 100% success - net withdrawal is 0


class TestGetCachedShillerDf:
    """Tests for get_cached_shiller_df function."""

    def test_caches_result(self):
        session_state = {}
        # First call loads data
        df1 = utils.get_cached_shiller_df(session_state)
        assert 'shiller_df' in session_state
        assert df1 is not None
        # Second call returns cached
        df2 = utils.get_cached_shiller_df(session_state)
        assert df1 is df2

    def test_returns_existing_cache(self):
        fake_df = pd.DataFrame({'test': [1, 2, 3]})
        session_state = {'shiller_df': fake_df}
        result = utils.get_cached_shiller_df(session_state)
        assert result is fake_df
