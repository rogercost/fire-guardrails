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
            portfolio_returns, 6, 100000.0, 0.0, monthly_cashflows, 0.0
        )
        assert result == 1.0  # 100% success

    def test_high_withdrawal_fails(self):
        # Flat returns with withdrawals exceeding value should fail
        portfolio_returns = np.array([1.0] * 12)  # No growth
        monthly_cashflows = np.zeros(6, dtype=np.float64)
        # Withdraw more than initial value over 6 months
        result = utils.test_all_periods(
            portfolio_returns, 6, 100000.0, 20000.0, monthly_cashflows, 0.0
        )
        assert result == 0.0  # 0% success

    def test_cashflows_offset_withdrawals(self):
        # Cashflows that fully offset withdrawals should succeed
        portfolio_returns = np.array([1.0] * 12)
        monthly_cashflows = np.array([5000.0] * 6, dtype=np.float64)
        result = utils.test_all_periods(
            portfolio_returns, 6, 100000.0, 5000.0, monthly_cashflows, 0.0
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


class TestEvaluateConditionalCashflows:
    """Tests for evaluate_conditional_cashflows function."""

    def test_empty_configs(self):
        result = utils.evaluate_conditional_cashflows(
            spending_target=400.0,
            initial_spending=1000.0,
            month_idx=0,
            conditional_configs=[],
            one_time_triggered={},
            recurring_state={},
        )
        assert result == 0.0

    def test_one_time_triggers_and_applies_next_month(self):
        configs = [{"cashflow_type": "one_time", "trigger_threshold_multiplier": 0.5, "amount": 1000.0}]
        triggered = {0: None}
        recurring = {}

        # Month 0: spending at 40% of initial - triggers
        result = utils.evaluate_conditional_cashflows(
            spending_target=400.0,
            initial_spending=1000.0,
            month_idx=0,
            conditional_configs=configs,
            one_time_triggered=triggered,
            recurring_state=recurring,
        )
        assert result == 0.0  # No contribution in trigger month
        assert triggered[0] == 0  # Marked as triggered in month 0

        # Month 1: cashflow applied
        result = utils.evaluate_conditional_cashflows(
            spending_target=400.0,
            initial_spending=1000.0,
            month_idx=1,
            conditional_configs=configs,
            one_time_triggered=triggered,
            recurring_state=recurring,
        )
        assert result == 1000.0  # One-time contribution

        # Month 2: no more contribution
        result = utils.evaluate_conditional_cashflows(
            spending_target=400.0,
            initial_spending=1000.0,
            month_idx=2,
            conditional_configs=configs,
            one_time_triggered=triggered,
            recurring_state=recurring,
        )
        assert result == 0.0

    def test_one_time_does_not_trigger_above_threshold(self):
        configs = [{"cashflow_type": "one_time", "trigger_threshold_multiplier": 0.5, "amount": 1000.0}]
        triggered = {0: None}
        recurring = {}

        # Month 0: spending at 60% - above 50% threshold
        result = utils.evaluate_conditional_cashflows(
            spending_target=600.0,
            initial_spending=1000.0,
            month_idx=0,
            conditional_configs=configs,
            one_time_triggered=triggered,
            recurring_state=recurring,
        )
        assert result == 0.0
        assert triggered[0] is None  # Not triggered

    def test_recurring_activates_and_deactivates(self):
        configs = [{"cashflow_type": "recurring", "trigger_threshold_multiplier": 0.6, "amount": 500.0}]
        triggered = {}
        recurring = {0: {"active": False, "started_month": None}}

        # Month 0: spending at 50% - below 60% threshold, activates
        result = utils.evaluate_conditional_cashflows(
            spending_target=500.0,
            initial_spending=1000.0,
            month_idx=0,
            conditional_configs=configs,
            one_time_triggered=triggered,
            recurring_state=recurring,
        )
        assert result == 0.0  # No contribution in activation month
        assert recurring[0]["active"] is True
        assert recurring[0]["started_month"] == 0

        # Month 1: still at 50%, cashflow applies
        result = utils.evaluate_conditional_cashflows(
            spending_target=500.0,
            initial_spending=1000.0,
            month_idx=1,
            conditional_configs=configs,
            one_time_triggered=triggered,
            recurring_state=recurring,
        )
        assert result == 500.0
        assert recurring[0]["active"] is True

        # Month 2: recovered to 100%, deactivates
        result = utils.evaluate_conditional_cashflows(
            spending_target=1000.0,
            initial_spending=1000.0,
            month_idx=2,
            conditional_configs=configs,
            one_time_triggered=triggered,
            recurring_state=recurring,
        )
        assert result == 0.0
        assert recurring[0]["active"] is False

    def test_recurring_can_reactivate(self):
        configs = [{"cashflow_type": "recurring", "trigger_threshold_multiplier": 0.5, "amount": 300.0}]
        triggered = {}
        recurring = {0: {"active": False, "started_month": None}}

        # Month 0: activate
        utils.evaluate_conditional_cashflows(400.0, 1000.0, 0, configs, triggered, recurring)
        assert recurring[0]["active"] is True

        # Month 1: contribute
        result = utils.evaluate_conditional_cashflows(400.0, 1000.0, 1, configs, triggered, recurring)
        assert result == 300.0

        # Month 2: recover to 100%
        utils.evaluate_conditional_cashflows(1000.0, 1000.0, 2, configs, triggered, recurring)
        assert recurring[0]["active"] is False

        # Month 3: still at 100%
        result = utils.evaluate_conditional_cashflows(1000.0, 1000.0, 3, configs, triggered, recurring)
        assert result == 0.0

        # Month 4: drop again, reactivate
        utils.evaluate_conditional_cashflows(400.0, 1000.0, 4, configs, triggered, recurring)
        assert recurring[0]["active"] is True
        assert recurring[0]["started_month"] == 4

        # Month 5: contribute again
        result = utils.evaluate_conditional_cashflows(400.0, 1000.0, 5, configs, triggered, recurring)
        assert result == 300.0

    def test_multiple_conditional_cashflows(self):
        configs = [
            {"cashflow_type": "one_time", "trigger_threshold_multiplier": 0.5, "amount": 2000.0},
            {"cashflow_type": "recurring", "trigger_threshold_multiplier": 0.7, "amount": 300.0},
        ]
        one_time_triggered = {0: 0}  # One-time already triggered in month 0
        recurring_state = {1: {"active": True, "started_month": 0}}  # Recurring already active

        # Month 1: one-time contributes (first month after trigger), recurring contributes
        result = utils.evaluate_conditional_cashflows(
            spending_target=400.0,
            initial_spending=1000.0,
            month_idx=1,
            conditional_configs=configs,
            one_time_triggered=one_time_triggered,
            recurring_state=recurring_state,
        )
        assert result == 2300.0  # 2000 + 300

    def test_threshold_exactly_matched_does_not_trigger(self):
        configs = [{"cashflow_type": "recurring", "trigger_threshold_multiplier": 0.5, "amount": 100.0}]
        recurring = {0: {"active": False, "started_month": None}}

        # Spending exactly at threshold (50% == 50%)
        result = utils.evaluate_conditional_cashflows(
            spending_target=500.0,
            initial_spending=1000.0,
            month_idx=0,
            conditional_configs=configs,
            one_time_triggered={},
            recurring_state=recurring,
        )
        assert recurring[0]["active"] is False  # Not triggered - must be strictly below

    def test_recovery_exactly_at_100_percent_deactivates(self):
        configs = [{"cashflow_type": "recurring", "trigger_threshold_multiplier": 0.5, "amount": 100.0}]
        recurring = {0: {"active": True, "started_month": 0}}

        # Recovery at exactly 100%
        result = utils.evaluate_conditional_cashflows(
            spending_target=1000.0,
            initial_spending=1000.0,
            month_idx=5,
            conditional_configs=configs,
            one_time_triggered={},
            recurring_state=recurring,
        )
        assert recurring[0]["active"] is False  # 100% >= 100%, deactivated

    def test_zero_initial_spending_returns_zero(self):
        configs = [{"cashflow_type": "recurring", "trigger_threshold_multiplier": 0.5, "amount": 100.0}]
        recurring = {0: {"active": False, "started_month": None}}

        result = utils.evaluate_conditional_cashflows(
            spending_target=0.0,
            initial_spending=0.0,
            month_idx=0,
            conditional_configs=configs,
            one_time_triggered={},
            recurring_state=recurring,
        )
        assert result == 0.0  # Graceful handling
