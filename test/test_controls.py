"""Tests for controls.py helper functions."""

import datetime
import pytest
from unittest.mock import MagicMock, patch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from controls import sanitize_cashflows, cashflows_to_tuple


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
