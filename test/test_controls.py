"""Tests for controls.py helper functions."""

import datetime
import pytest
from unittest.mock import MagicMock, patch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from controls import (
    sanitize_cashflows,
    cashflows_to_tuple,
    sanitize_conditional_cashflows,
    CONDITIONAL_THRESHOLD_OPTIONS,
    CONDITIONAL_TYPE_OPTIONS,
)


class TestSanitizeCashflows:
    """Tests for sanitize_cashflows function."""

    def test_empty_list(self):
        assert sanitize_cashflows([]) == []

    def test_none_input(self):
        assert sanitize_cashflows(None) == []

    def test_valid_cashflow(self):
        raw = [{"start_month": 0, "end_month": 12, "amount": 1000.0, "label": "Test"}]
        result = sanitize_cashflows(raw)
        assert len(result) == 1
        assert result[0]["start_month"] == 0
        assert result[0]["end_month"] == 12
        assert result[0]["amount"] == 1000.0
        assert result[0]["label"] == "Test"

    def test_multiple_cashflows(self):
        raw = [
            {"start_month": 0, "end_month": 12, "amount": 1000.0},
            {"start_month": 6, "end_month": 18, "amount": 500.0},
        ]
        result = sanitize_cashflows(raw)
        assert len(result) == 2

    def test_filters_invalid_end_before_start(self):
        raw = [
            {"start_month": 12, "end_month": 0, "amount": 1000.0},  # Invalid
            {"start_month": 0, "end_month": 12, "amount": 500.0},  # Valid
        ]
        result = sanitize_cashflows(raw)
        assert len(result) == 1
        assert result[0]["amount"] == 500.0

    def test_converts_string_numbers(self):
        raw = [{"start_month": "0", "end_month": "12", "amount": "1000.0"}]
        result = sanitize_cashflows(raw)
        assert len(result) == 1
        assert result[0]["start_month"] == 0
        assert isinstance(result[0]["start_month"], int)

    def test_handles_missing_label(self):
        raw = [{"start_month": 0, "end_month": 12, "amount": 1000.0}]
        result = sanitize_cashflows(raw)
        assert result[0]["label"] is None

    def test_strips_label_whitespace(self):
        raw = [{"start_month": 0, "end_month": 12, "amount": 1000.0, "label": "  Test  "}]
        result = sanitize_cashflows(raw)
        assert result[0]["label"] == "Test"

    def test_skips_invalid_types(self):
        raw = [
            {"start_month": "invalid", "end_month": 12, "amount": 1000.0},
            {"start_month": 0, "end_month": 12, "amount": 500.0},
        ]
        result = sanitize_cashflows(raw)
        assert len(result) == 1

    def test_skips_non_dict_items(self):
        raw = [
            "not a dict",
            {"start_month": 0, "end_month": 12, "amount": 500.0},
        ]
        result = sanitize_cashflows(raw)
        assert len(result) == 1

    def test_negative_amount_allowed(self):
        raw = [{"start_month": 0, "end_month": 12, "amount": -500.0}]
        result = sanitize_cashflows(raw)
        assert len(result) == 1
        assert result[0]["amount"] == -500.0

    def test_zero_amount_allowed(self):
        raw = [{"start_month": 0, "end_month": 12, "amount": 0.0}]
        result = sanitize_cashflows(raw)
        assert len(result) == 1
        assert result[0]["amount"] == 0.0

    def test_same_start_and_end_month_allowed(self):
        raw = [{"start_month": 5, "end_month": 5, "amount": 1000.0}]
        result = sanitize_cashflows(raw)
        assert len(result) == 1


class TestCashflowsToTuple:
    """Tests for cashflows_to_tuple function."""

    def test_empty_list(self):
        assert cashflows_to_tuple([]) == ()

    def test_single_cashflow(self):
        cashflows = [{"start_month": 0, "end_month": 12, "amount": 1000.0}]
        result = cashflows_to_tuple(cashflows)
        assert result == ((0, 12, 1000.0),)

    def test_multiple_cashflows(self):
        cashflows = [
            {"start_month": 0, "end_month": 12, "amount": 1000.0},
            {"start_month": 6, "end_month": 18, "amount": 500.0},
        ]
        result = cashflows_to_tuple(cashflows)
        assert result == ((0, 12, 1000.0), (6, 18, 500.0))

    def test_excludes_label(self):
        cashflows = [{"start_month": 0, "end_month": 12, "amount": 1000.0, "label": "Test"}]
        result = cashflows_to_tuple(cashflows)
        assert len(result[0]) == 3  # Only start, end, amount

    def test_tuple_is_hashable(self):
        cashflows = [{"start_month": 0, "end_month": 12, "amount": 1000.0}]
        result = cashflows_to_tuple(cashflows)
        # Should be usable as dict key
        d = {result: "test"}
        assert d[result] == "test"

    def test_preserves_order(self):
        cashflows = [
            {"start_month": 10, "end_month": 20, "amount": 100.0},
            {"start_month": 0, "end_month": 5, "amount": 200.0},
        ]
        result = cashflows_to_tuple(cashflows)
        assert result[0] == (10, 20, 100.0)
        assert result[1] == (0, 5, 200.0)


class TestGetStateHelpers:
    """Tests for get_date_state, get_int_state, get_float_state using mocked session_state."""

    @pytest.fixture
    def mock_session_state(self):
        """Create a mock session_state dict."""
        return {}

    def test_get_date_state_returns_date(self, mock_session_state):
        mock_session_state["my_date"] = datetime.date(2020, 1, 1)
        with patch("controls.st") as mock_st:
            mock_st.session_state = mock_session_state
            from controls import get_date_state
            result = get_date_state("my_date", datetime.date(1999, 1, 1))
            assert result == datetime.date(2020, 1, 1)

    def test_get_date_state_returns_default_when_missing(self, mock_session_state):
        with patch("controls.st") as mock_st:
            mock_st.session_state = mock_session_state
            from controls import get_date_state
            default = datetime.date(1999, 1, 1)
            result = get_date_state("missing_key", default)
            assert result == default

    def test_get_date_state_parses_string(self, mock_session_state):
        mock_session_state["my_date"] = "2020-06-15"
        with patch("controls.st") as mock_st:
            mock_st.session_state = mock_session_state
            from controls import get_date_state
            result = get_date_state("my_date", datetime.date(1999, 1, 1))
            assert result == datetime.date(2020, 6, 15)

    def test_get_int_state_returns_int(self, mock_session_state):
        mock_session_state["count"] = 42
        with patch("controls.st") as mock_st:
            mock_st.session_state = mock_session_state
            from controls import get_int_state
            result = get_int_state("count", 0)
            assert result == 42

    def test_get_int_state_returns_default_when_missing(self, mock_session_state):
        with patch("controls.st") as mock_st:
            mock_st.session_state = mock_session_state
            from controls import get_int_state
            result = get_int_state("missing", 99)
            assert result == 99

    def test_get_float_state_returns_float(self, mock_session_state):
        mock_session_state["rate"] = 0.05
        with patch("controls.st") as mock_st:
            mock_st.session_state = mock_session_state
            from controls import get_float_state
            result = get_float_state("rate", 0.0)
            assert result == 0.05

    def test_get_float_state_returns_default_when_missing(self, mock_session_state):
        with patch("controls.st") as mock_st:
            mock_st.session_state = mock_session_state
            from controls import get_float_state
            result = get_float_state("missing", 0.99)
            assert result == 0.99


class TestConditionalCashflowConstants:
    """Tests for conditional cashflow constants."""

    def test_threshold_options_range(self):
        assert CONDITIONAL_THRESHOLD_OPTIONS[0] == "5%"
        assert CONDITIONAL_THRESHOLD_OPTIONS[-1] == "95%"
        assert len(CONDITIONAL_THRESHOLD_OPTIONS) == 19  # 5, 10, 15, ..., 95

    def test_type_options(self):
        assert CONDITIONAL_TYPE_OPTIONS == ("One-time", "Recurring")


class TestSanitizeConditionalCashflows:
    """Tests for sanitize_conditional_cashflows function."""

    def test_empty_list(self):
        assert sanitize_conditional_cashflows([]) == []

    def test_none_input(self):
        assert sanitize_conditional_cashflows(None) == []

    def test_valid_one_time_cashflow(self):
        raw = [{
            "cashflow_type": "one_time",
            "trigger_threshold": "50%",
            "amount": 10000.0,
            "label": "Sell house"
        }]
        result = sanitize_conditional_cashflows(raw)
        assert len(result) == 1
        assert result[0]["cashflow_type"] == "one_time"
        assert result[0]["trigger_threshold"] == "50%"
        assert result[0]["amount"] == 10000.0
        assert result[0]["label"] == "Sell house"

    def test_valid_recurring_cashflow(self):
        raw = [{
            "cashflow_type": "recurring",
            "trigger_threshold": "80%",
            "amount": 500.0,
            "label": "Part-time job"
        }]
        result = sanitize_conditional_cashflows(raw)
        assert len(result) == 1
        assert result[0]["cashflow_type"] == "recurring"
        assert result[0]["trigger_threshold"] == "80%"
        assert result[0]["amount"] == 500.0

    def test_multiple_cashflows(self):
        raw = [
            {"cashflow_type": "one_time", "trigger_threshold": "50%", "amount": 10000.0},
            {"cashflow_type": "recurring", "trigger_threshold": "80%", "amount": 500.0},
        ]
        result = sanitize_conditional_cashflows(raw)
        assert len(result) == 2

    def test_filters_invalid_cashflow_type(self):
        raw = [
            {"cashflow_type": "invalid_type", "trigger_threshold": "50%", "amount": 1000.0},
            {"cashflow_type": "one_time", "trigger_threshold": "50%", "amount": 500.0},
        ]
        result = sanitize_conditional_cashflows(raw)
        assert len(result) == 1
        assert result[0]["cashflow_type"] == "one_time"

    def test_default_cashflow_type(self):
        raw = [{"trigger_threshold": "50%", "amount": 1000.0}]
        result = sanitize_conditional_cashflows(raw)
        assert len(result) == 1
        assert result[0]["cashflow_type"] == "one_time"

    def test_default_threshold(self):
        raw = [{"cashflow_type": "one_time", "amount": 1000.0}]
        result = sanitize_conditional_cashflows(raw)
        assert result[0]["trigger_threshold"] == "50%"

    def test_handles_missing_label(self):
        raw = [{"cashflow_type": "one_time", "trigger_threshold": "50%", "amount": 1000.0}]
        result = sanitize_conditional_cashflows(raw)
        assert result[0]["label"] is None

    def test_strips_label_whitespace(self):
        raw = [{
            "cashflow_type": "one_time",
            "trigger_threshold": "50%",
            "amount": 1000.0,
            "label": "  Test Label  "
        }]
        result = sanitize_conditional_cashflows(raw)
        assert result[0]["label"] == "Test Label"

    def test_skips_invalid_amount(self):
        raw = [
            {"cashflow_type": "one_time", "trigger_threshold": "50%", "amount": "invalid"},
            {"cashflow_type": "one_time", "trigger_threshold": "50%", "amount": 500.0},
        ]
        result = sanitize_conditional_cashflows(raw)
        assert len(result) == 1
        assert result[0]["amount"] == 500.0

    def test_skips_non_dict_items(self):
        raw = [
            "not a dict",
            {"cashflow_type": "one_time", "trigger_threshold": "50%", "amount": 500.0},
        ]
        result = sanitize_conditional_cashflows(raw)
        assert len(result) == 1

    def test_negative_amount_allowed(self):
        raw = [{"cashflow_type": "one_time", "trigger_threshold": "50%", "amount": -500.0}]
        result = sanitize_conditional_cashflows(raw)
        assert len(result) == 1
        assert result[0]["amount"] == -500.0

    def test_converts_string_amount(self):
        raw = [{"cashflow_type": "one_time", "trigger_threshold": "50%", "amount": "1000.0"}]
        result = sanitize_conditional_cashflows(raw)
        assert result[0]["amount"] == 1000.0
        assert isinstance(result[0]["amount"], float)


class TestClearConditionalCashflowWidgetState:
    """Tests for clear_conditional_cashflow_widget_state function."""

    def test_clears_all_from_start(self):
        mock_state = {
            "ccf_type_0": "One-time",
            "ccf_threshold_0": "50%",
            "ccf_amount_0": 1000.0,
            "ccf_label_0": "Test",
            "ccf_type_1": "Recurring",
            "other_key": "preserved",
        }
        with patch("controls.st") as mock_st:
            mock_st.session_state = mock_state
            from controls import clear_conditional_cashflow_widget_state
            clear_conditional_cashflow_widget_state(0)
            assert "ccf_type_0" not in mock_state
            assert "ccf_threshold_0" not in mock_state
            assert "ccf_amount_0" not in mock_state
            assert "ccf_label_0" not in mock_state
            assert "ccf_type_1" not in mock_state
            assert mock_state["other_key"] == "preserved"

    def test_clears_from_specific_index(self):
        mock_state = {
            "ccf_type_0": "One-time",
            "ccf_type_1": "Recurring",
            "ccf_type_2": "One-time",
        }
        with patch("controls.st") as mock_st:
            mock_st.session_state = mock_state
            from controls import clear_conditional_cashflow_widget_state
            clear_conditional_cashflow_widget_state(1)
            assert "ccf_type_0" in mock_state
            assert "ccf_type_1" not in mock_state
            assert "ccf_type_2" not in mock_state

    def test_handles_empty_state(self):
        mock_state = {}
        with patch("controls.st") as mock_st:
            mock_st.session_state = mock_state
            from controls import clear_conditional_cashflow_widget_state
            clear_conditional_cashflow_widget_state(0)  # Should not raise


class TestEnsureConditionalCashflowWidgetState:
    """Tests for ensure_conditional_cashflow_widget_state function."""

    def test_sets_defaults_when_missing(self):
        mock_state = {}
        flow = {
            "cashflow_type": "one_time",
            "trigger_threshold": "75%",
            "amount": 5000.0,
            "label": "Test Flow"
        }
        with patch("controls.st") as mock_st:
            mock_st.session_state = mock_state
            from controls import ensure_conditional_cashflow_widget_state
            ensure_conditional_cashflow_widget_state(0, flow)
            assert mock_state["ccf_type_0"] == "One-time"
            assert mock_state["ccf_threshold_0"] == "75%"
            assert mock_state["ccf_amount_0"] == 5000.0
            assert mock_state["ccf_label_0"] == "Test Flow"

    def test_converts_recurring_type(self):
        mock_state = {}
        flow = {"cashflow_type": "recurring", "trigger_threshold": "50%", "amount": 500.0}
        with patch("controls.st") as mock_st:
            mock_st.session_state = mock_state
            from controls import ensure_conditional_cashflow_widget_state
            ensure_conditional_cashflow_widget_state(0, flow)
            assert mock_state["ccf_type_0"] == "Recurring"

    def test_preserves_existing_values(self):
        mock_state = {
            "ccf_type_0": "Recurring",
            "ccf_threshold_0": "25%",
            "ccf_amount_0": 2000.0,
            "ccf_label_0": "Existing Label"
        }
        flow = {
            "cashflow_type": "one_time",
            "trigger_threshold": "75%",
            "amount": 5000.0,
            "label": "New Label"
        }
        with patch("controls.st") as mock_st:
            mock_st.session_state = mock_state
            from controls import ensure_conditional_cashflow_widget_state
            ensure_conditional_cashflow_widget_state(0, flow)
            assert mock_state["ccf_type_0"] == "Recurring"
            assert mock_state["ccf_threshold_0"] == "25%"
            assert mock_state["ccf_amount_0"] == 2000.0
            assert mock_state["ccf_label_0"] == "Existing Label"

    def test_default_label_when_missing(self):
        mock_state = {}
        flow = {"cashflow_type": "one_time", "trigger_threshold": "50%", "amount": 1000.0}
        with patch("controls.st") as mock_st:
            mock_st.session_state = mock_state
            from controls import ensure_conditional_cashflow_widget_state
            ensure_conditional_cashflow_widget_state(2, flow)
            assert mock_state["ccf_label_2"] == "Conditional 3"

    def test_replaces_empty_label(self):
        mock_state = {"ccf_label_0": "   "}
        flow = {"cashflow_type": "one_time", "trigger_threshold": "50%", "amount": 1000.0}
        with patch("controls.st") as mock_st:
            mock_st.session_state = mock_state
            from controls import ensure_conditional_cashflow_widget_state
            ensure_conditional_cashflow_widget_state(0, flow)
            assert mock_state["ccf_label_0"] == "Conditional 1"


class TestSyncConditionalCashflowsFromWidgets:
    """Tests for sync_conditional_cashflows_from_widgets function."""

    def test_syncs_widget_values_to_flows(self):
        flows = [
            {"cashflow_type": "one_time", "trigger_threshold": "50%", "amount": 1000.0, "label": "Old"}
        ]
        mock_state = {
            "conditional_cashflows": flows,
            "ccf_type_0": "Recurring",
            "ccf_threshold_0": "75%",
            "ccf_amount_0": 2000.0,
            "ccf_label_0": "Updated"
        }
        with patch("controls.st") as mock_st:
            mock_st.session_state = mock_state
            from controls import sync_conditional_cashflows_from_widgets
            sync_conditional_cashflows_from_widgets()
            assert flows[0]["cashflow_type"] == "recurring"
            assert flows[0]["trigger_threshold"] == "75%"
            assert flows[0]["amount"] == 2000.0
            assert flows[0]["label"] == "Updated"

    def test_handles_empty_flows(self):
        mock_state = {"conditional_cashflows": []}
        with patch("controls.st") as mock_st:
            mock_st.session_state = mock_state
            from controls import sync_conditional_cashflows_from_widgets
            sync_conditional_cashflows_from_widgets()  # Should not raise

    def test_handles_missing_flows(self):
        mock_state = {}
        with patch("controls.st") as mock_st:
            mock_st.session_state = mock_state
            from controls import sync_conditional_cashflows_from_widgets
            sync_conditional_cashflows_from_widgets()  # Should not raise

    def test_handles_invalid_amount(self):
        flows = [
            {"cashflow_type": "one_time", "trigger_threshold": "50%", "amount": 1000.0, "label": "Test"}
        ]
        mock_state = {
            "conditional_cashflows": flows,
            "ccf_type_0": "One-time",
            "ccf_threshold_0": "50%",
            "ccf_amount_0": "invalid",
            "ccf_label_0": "Test"
        }
        with patch("controls.st") as mock_st:
            mock_st.session_state = mock_state
            from controls import sync_conditional_cashflows_from_widgets
            sync_conditional_cashflows_from_widgets()
            assert flows[0]["amount"] == 1000.0  # Falls back to original

    def test_uses_defaults_for_missing_widget_values(self):
        flows = [
            {"cashflow_type": "one_time", "trigger_threshold": "50%", "amount": 1000.0, "label": "Test"}
        ]
        mock_state = {"conditional_cashflows": flows}
        with patch("controls.st") as mock_st:
            mock_st.session_state = mock_state
            from controls import sync_conditional_cashflows_from_widgets
            sync_conditional_cashflows_from_widgets()
            assert flows[0]["cashflow_type"] == "one_time"
            assert flows[0]["trigger_threshold"] == "50%"

    def test_replaces_empty_label_with_default(self):
        flows = [
            {"cashflow_type": "one_time", "trigger_threshold": "50%", "amount": 1000.0, "label": "Old"}
        ]
        mock_state = {
            "conditional_cashflows": flows,
            "ccf_type_0": "One-time",
            "ccf_threshold_0": "50%",
            "ccf_amount_0": 1000.0,
            "ccf_label_0": "   "
        }
        with patch("controls.st") as mock_st:
            mock_st.session_state = mock_state
            from controls import sync_conditional_cashflows_from_widgets
            sync_conditional_cashflows_from_widgets()
            assert flows[0]["label"] == "Conditional 1"
