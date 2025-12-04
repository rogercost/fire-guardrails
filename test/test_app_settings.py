"""Tests for app_settings.py serialization and settings management."""

import datetime
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app_settings import CashflowSetting, Settings


class TestCashflowSetting:
    """Tests for CashflowSetting dataclass."""

    def test_from_dict_valid(self):
        data = {"start_month": 0, "end_month": 12, "amount": 1000.0, "label": "Social Security"}
        result = CashflowSetting.from_dict(data)
        assert result is not None
        assert result.start_month == 0
        assert result.end_month == 12
        assert result.amount == 1000.0
        assert result.label == "Social Security"

    def test_from_dict_missing_label(self):
        data = {"start_month": 0, "end_month": 12, "amount": 1000.0}
        result = CashflowSetting.from_dict(data)
        assert result is not None
        assert result.label == ""

    def test_from_dict_invalid_end_before_start(self):
        data = {"start_month": 12, "end_month": 0, "amount": 1000.0}
        result = CashflowSetting.from_dict(data)
        assert result is None

    def test_from_dict_non_dict_input(self):
        result = CashflowSetting.from_dict("not a dict")
        assert result is None

    def test_from_dict_invalid_types(self):
        data = {"start_month": "invalid", "end_month": 12, "amount": 1000.0}
        result = CashflowSetting.from_dict(data)
        assert result is None

    def test_to_serializable_roundtrip(self):
        original = CashflowSetting(start_month=0, end_month=12, amount=1000.0, label="Test")
        serialized = original.to_serializable()
        restored = CashflowSetting.from_dict(serialized)
        assert restored.start_month == original.start_month
        assert restored.end_month == original.end_month
        assert restored.amount == original.amount
        assert restored.label == original.label

    def test_to_calculation_dict(self):
        cf = CashflowSetting(start_month=0, end_month=12, amount=1000.0, label="Test")
        calc_dict = cf.to_calculation_dict()
        assert "label" not in calc_dict
        assert calc_dict["start_month"] == 0
        assert calc_dict["end_month"] == 12
        assert calc_dict["amount"] == 1000.0

    def test_signature(self):
        cf = CashflowSetting(start_month=0, end_month=12, amount=1000.0, label="Test")
        sig = cf.signature()
        assert sig == (0, 12, 1000.0)


class TestSettings:
    """Tests for Settings dataclass."""

    @pytest.fixture
    def default_settings(self):
        return Settings(
            mode="Simulation Mode",
            start_date=datetime.date(2020, 1, 1),
            retirement_duration_months=360,
            analysis_start_date=datetime.date(1871, 1, 1),
            initial_value=1_000_000.0,
            stock_pct=0.75,
            target_success_rate=0.90,
            initial_monthly_spending=4000.0,
            initial_spending_overridden=False,
            upper_guardrail_success=1.0,
            lower_guardrail_success=0.75,
            upper_adjustment_fraction=1.0,
            lower_adjustment_fraction=0.1,
            adjustment_threshold=0.05,
            adjustment_frequency="Monthly",
            spending_cap_option="Unlimited",
            spending_floor_option="Unlimited",
            cashflows=[],
        )

    def test_to_dict_roundtrip(self, default_settings):
        serialized = default_settings.to_dict()
        restored = Settings.from_dict(serialized)
        assert restored.mode == default_settings.mode
        assert restored.start_date == default_settings.start_date
        assert restored.retirement_duration_months == default_settings.retirement_duration_months
        assert restored.initial_value == default_settings.initial_value
        assert restored.stock_pct == default_settings.stock_pct

    def test_to_base64_roundtrip(self, default_settings):
        encoded = default_settings.to_base64()
        restored = Settings.from_base64(encoded)
        assert restored.mode == default_settings.mode
        assert restored.start_date == default_settings.start_date
        assert restored.initial_value == default_settings.initial_value

    def test_base64_is_url_safe(self, default_settings):
        encoded = default_settings.to_base64()
        # URL-safe base64 should not contain +, /, or =
        assert "+" not in encoded
        assert "/" not in encoded
        # Padding is stripped
        assert not encoded.endswith("=")

    def test_spending_cap_multiplier_unlimited(self, default_settings):
        assert default_settings.spending_cap_multiplier is None

    def test_spending_cap_multiplier_percentage(self):
        settings = Settings(
            mode="Simulation Mode",
            start_date=datetime.date(2020, 1, 1),
            retirement_duration_months=360,
            analysis_start_date=datetime.date(1871, 1, 1),
            initial_value=1_000_000.0,
            stock_pct=0.75,
            target_success_rate=0.90,
            initial_monthly_spending=4000.0,
            initial_spending_overridden=False,
            upper_guardrail_success=1.0,
            lower_guardrail_success=0.75,
            upper_adjustment_fraction=1.0,
            lower_adjustment_fraction=0.1,
            adjustment_threshold=0.05,
            adjustment_frequency="Monthly",
            spending_cap_option="150%",
            spending_floor_option="Unlimited",
            cashflows=[],
        )
        assert settings.spending_cap_multiplier == 1.5

    def test_spending_floor_multiplier_percentage(self):
        settings = Settings(
            mode="Simulation Mode",
            start_date=datetime.date(2020, 1, 1),
            retirement_duration_months=360,
            analysis_start_date=datetime.date(1871, 1, 1),
            initial_value=1_000_000.0,
            stock_pct=0.75,
            target_success_rate=0.90,
            initial_monthly_spending=4000.0,
            initial_spending_overridden=False,
            upper_guardrail_success=1.0,
            lower_guardrail_success=0.75,
            upper_adjustment_fraction=1.0,
            lower_adjustment_fraction=0.1,
            adjustment_threshold=0.05,
            adjustment_frequency="Monthly",
            spending_cap_option="Unlimited",
            spending_floor_option="80%",
            cashflows=[],
        )
        assert settings.spending_floor_multiplier == 0.8

    def test_retirement_end_date(self, default_settings):
        # 360 months from 2020-01-01 should end at 2049-12-01
        end_date = default_settings.retirement_end_date()
        assert end_date == datetime.date(2049, 12, 1)

    def test_retirement_end_date_zero_months_raises(self):
        with pytest.raises(ValueError) as exc_info:
            Settings(
                mode="Simulation Mode",
                start_date=datetime.date(2020, 1, 1),
                retirement_duration_months=0,
                analysis_start_date=datetime.date(1871, 1, 1),
                initial_value=1_000_000.0,
                stock_pct=0.75,
                target_success_rate=0.90,
                initial_monthly_spending=4000.0,
                initial_spending_overridden=False,
                upper_guardrail_success=1.0,
                lower_guardrail_success=0.75,
                upper_adjustment_fraction=1.0,
                lower_adjustment_fraction=0.1,
                adjustment_threshold=0.05,
                adjustment_frequency="Monthly",
                spending_cap_option="Unlimited",
                spending_floor_option="Unlimited",
                cashflows=[],
            )
        assert "Retirement duration must be positive" in str(exc_info.value)

    def test_cashflows_preserved_in_roundtrip(self):
        cashflows = [
            CashflowSetting(start_month=0, end_month=120, amount=2000.0, label="SS"),
            CashflowSetting(start_month=60, end_month=180, amount=500.0, label="Pension"),
        ]
        settings = Settings(
            mode="Simulation Mode",
            start_date=datetime.date(2020, 1, 1),
            retirement_duration_months=360,
            analysis_start_date=datetime.date(1871, 1, 1),
            initial_value=1_000_000.0,
            stock_pct=0.75,
            target_success_rate=0.90,
            initial_monthly_spending=4000.0,
            initial_spending_overridden=False,
            upper_guardrail_success=1.0,
            lower_guardrail_success=0.75,
            upper_adjustment_fraction=1.0,
            lower_adjustment_fraction=0.1,
            adjustment_threshold=0.05,
            adjustment_frequency="Monthly",
            spending_cap_option="Unlimited",
            spending_floor_option="Unlimited",
            cashflows=cashflows,
        )
        encoded = settings.to_base64()
        restored = Settings.from_base64(encoded)
        assert len(restored.cashflows) == 2
        assert restored.cashflows[0].amount == 2000.0
        assert restored.cashflows[1].label == "Pension"

    def test_simulation_signature_changes_with_inputs(self, default_settings):
        sig1 = default_settings.simulation_signature()

        # Change a parameter
        modified = Settings(
            mode=default_settings.mode,
            start_date=default_settings.start_date,
            retirement_duration_months=default_settings.retirement_duration_months,
            analysis_start_date=default_settings.analysis_start_date,
            initial_value=2_000_000.0,  # Changed
            stock_pct=default_settings.stock_pct,
            target_success_rate=default_settings.target_success_rate,
            initial_monthly_spending=default_settings.initial_monthly_spending,
            initial_spending_overridden=default_settings.initial_spending_overridden,
            upper_guardrail_success=default_settings.upper_guardrail_success,
            lower_guardrail_success=default_settings.lower_guardrail_success,
            upper_adjustment_fraction=default_settings.upper_adjustment_fraction,
            lower_adjustment_fraction=default_settings.lower_adjustment_fraction,
            adjustment_threshold=default_settings.adjustment_threshold,
            adjustment_frequency=default_settings.adjustment_frequency,
            spending_cap_option=default_settings.spending_cap_option,
            spending_floor_option=default_settings.spending_floor_option,
            cashflows=[],
        )
        sig2 = modified.simulation_signature()
        assert sig1 != sig2

    def test_to_isr_params(self, default_settings):
        params = default_settings.to_isr_params()
        assert params["start_date"] == default_settings.start_date
        assert params["duration_months"] == 360
        assert params["initial_value"] == 1_000_000.0
        assert params["stock_pct"] == 0.75
        assert params["desired_success_rate"] == 0.90

    def test_to_guardrail_params(self, default_settings):
        params = default_settings.to_guardrail_params()
        assert params["upper_sr"] == 1.0
        assert params["lower_sr"] == 0.75
        assert params["initial_spending"] == 4000.0


class TestRelativeOptionToMultiplier:
    """Tests for _relative_option_to_multiplier helper."""

    def test_unlimited_returns_none(self):
        from app_settings import _relative_option_to_multiplier
        assert _relative_option_to_multiplier("Unlimited") is None
        assert _relative_option_to_multiplier("unlimited") is None
        assert _relative_option_to_multiplier("UNLIMITED") is None

    def test_percentage_string(self):
        from app_settings import _relative_option_to_multiplier
        assert _relative_option_to_multiplier("100%") == 1.0
        assert _relative_option_to_multiplier("150%") == 1.5
        assert _relative_option_to_multiplier("50%") == 0.5

    def test_percentage_with_whitespace(self):
        from app_settings import _relative_option_to_multiplier
        assert _relative_option_to_multiplier("  100%  ") == 1.0

    def test_none_returns_none(self):
        from app_settings import _relative_option_to_multiplier
        assert _relative_option_to_multiplier(None) is None

    def test_invalid_string_returns_none(self):
        from app_settings import _relative_option_to_multiplier
        assert _relative_option_to_multiplier("invalid") is None


class TestSettingsValidation:
    """Tests for Settings validation in __post_init__."""

    def _make_settings(self, **overrides):
        """Helper to create Settings with defaults, overriding specific fields."""
        defaults = {
            "mode": "Simulation Mode",
            "start_date": datetime.date(2020, 1, 1),
            "retirement_duration_months": 360,
            "analysis_start_date": datetime.date(1871, 1, 1),
            "initial_value": 1_000_000.0,
            "stock_pct": 0.75,
            "target_success_rate": 0.90,
            "initial_monthly_spending": 4000.0,
            "initial_spending_overridden": False,
            "upper_guardrail_success": 1.0,
            "lower_guardrail_success": 0.75,
            "upper_adjustment_fraction": 1.0,
            "lower_adjustment_fraction": 0.1,
            "adjustment_threshold": 0.05,
            "adjustment_frequency": "Monthly",
            "spending_cap_option": "Unlimited",
            "spending_floor_option": "Unlimited",
            "cashflows": [],
        }
        defaults.update(overrides)
        return Settings(**defaults)

    def test_negative_initial_value_raises(self):
        with pytest.raises(ValueError) as exc_info:
            self._make_settings(initial_value=-100000.0)
        assert "Initial portfolio value must be positive" in str(exc_info.value)

    def test_zero_initial_value_raises(self):
        with pytest.raises(ValueError) as exc_info:
            self._make_settings(initial_value=0.0)
        assert "Initial portfolio value must be positive" in str(exc_info.value)

    def test_negative_monthly_spending_raises(self):
        with pytest.raises(ValueError) as exc_info:
            self._make_settings(initial_monthly_spending=-1000.0)
        assert "Initial monthly spending cannot be negative" in str(exc_info.value)

    def test_zero_monthly_spending_allowed(self):
        settings = self._make_settings(initial_monthly_spending=0.0)
        assert settings.initial_monthly_spending == 0.0

    def test_negative_retirement_duration_raises(self):
        with pytest.raises(ValueError) as exc_info:
            self._make_settings(retirement_duration_months=-12)
        assert "Retirement duration must be positive" in str(exc_info.value)

    def test_stock_pct_above_one_raises(self):
        with pytest.raises(ValueError) as exc_info:
            self._make_settings(stock_pct=1.5)
        assert "Stock percentage must be between 0% and 100%" in str(exc_info.value)

    def test_stock_pct_negative_raises(self):
        with pytest.raises(ValueError) as exc_info:
            self._make_settings(stock_pct=-0.1)
        assert "Stock percentage must be between 0% and 100%" in str(exc_info.value)

    def test_stock_pct_boundary_values_allowed(self):
        settings_zero = self._make_settings(stock_pct=0.0)
        assert settings_zero.stock_pct == 0.0
        settings_one = self._make_settings(stock_pct=1.0)
        assert settings_one.stock_pct == 1.0

    def test_target_success_rate_above_one_raises(self):
        with pytest.raises(ValueError) as exc_info:
            self._make_settings(target_success_rate=1.1)
        assert "Target success rate must be between 0% and 100%" in str(exc_info.value)

    def test_analysis_start_after_retirement_start_raises(self):
        with pytest.raises(ValueError) as exc_info:
            self._make_settings(
                start_date=datetime.date(2000, 1, 1),
                analysis_start_date=datetime.date(2010, 1, 1),
            )
        assert "Historical analysis start date" in str(exc_info.value)
        assert "cannot be after" in str(exc_info.value)

    def test_analysis_start_equals_retirement_start_allowed(self):
        settings = self._make_settings(
            start_date=datetime.date(2000, 1, 1),
            analysis_start_date=datetime.date(2000, 1, 1),
        )
        assert settings.start_date == settings.analysis_start_date


class TestSettingsBase64ErrorHandling:
    """Tests for user-friendly error messages from from_base64."""

    def test_corrupted_base64_gives_friendly_error(self):
        with pytest.raises(ValueError) as exc_info:
            Settings.from_base64("!!!not-valid-base64!!!")
        # May fail at base64 decode or gzip decompress depending on input
        error_msg = str(exc_info.value).lower()
        assert "corrupted" in error_msg or "invalid" in error_msg

    def test_truncated_data_gives_friendly_error(self):
        # Valid base64 but not valid gzip
        with pytest.raises(ValueError) as exc_info:
            Settings.from_base64("SGVsbG8gV29ybGQ")  # "Hello World" in base64
        assert "could not be decompressed" in str(exc_info.value)

    def test_invalid_json_gives_friendly_error(self):
        import gzip
        import base64
        # Gzip some invalid JSON
        invalid_json = b"not valid json {"
        compressed = gzip.compress(invalid_json)
        encoded = base64.urlsafe_b64encode(compressed).decode("utf-8").rstrip("=")
        with pytest.raises(ValueError) as exc_info:
            Settings.from_base64(encoded)
        assert "invalid data" in str(exc_info.value)

    def test_non_dict_json_gives_friendly_error(self):
        import gzip
        import base64
        # Gzip a JSON array instead of object
        array_json = b'[1, 2, 3]'
        compressed = gzip.compress(array_json)
        encoded = base64.urlsafe_b64encode(compressed).decode("utf-8").rstrip("=")
        with pytest.raises(ValueError) as exc_info:
            Settings.from_base64(encoded)
        assert "format is invalid" in str(exc_info.value)
