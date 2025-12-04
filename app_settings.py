from __future__ import annotations

import base64
import datetime as dt
import gzip
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


def _relative_option_to_multiplier(option: Optional[str]) -> Optional[float]:
    """Translate a relative spending selection into a multiplier when possible."""

    if option is None:
        return None
    option = option.strip()
    if option.lower() == "unlimited":
        return None
    try:
        return float(option.strip("%")) / 100.0
    except (TypeError, ValueError):
        return None


@dataclass
class CashflowSetting:
    start_month: int
    end_month: int
    amount: float
    label: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Optional["CashflowSetting"]:
        if not isinstance(data, dict):
            return None
        try:
            start = int(data.get("start_month", 0))
            end = int(data.get("end_month", 0))
            amount = float(data.get("amount", 0.0))
        except (TypeError, ValueError):
            return None
        if end < start:
            return None
        label_raw = data.get("label")
        label = str(label_raw).strip() if label_raw is not None else ""
        return cls(start_month=start, end_month=end, amount=amount, label=label)

    def to_serializable(self) -> Dict[str, Any]:
        return {
            "start_month": int(self.start_month),
            "end_month": int(self.end_month),
            "amount": float(self.amount),
            "label": self.label,
        }

    def to_calculation_dict(self) -> Dict[str, Any]:
        return {
            "start_month": int(self.start_month),
            "end_month": int(self.end_month),
            "amount": float(self.amount),
        }

    def signature(self) -> tuple:
        return (int(self.start_month), int(self.end_month), float(self.amount))


@dataclass
class Settings:
    mode: str
    start_date: dt.date
    retirement_duration_months: int
    analysis_start_date: dt.date
    initial_value: float
    stock_pct: float
    target_success_rate: float
    initial_monthly_spending: float
    initial_spending_overridden: bool
    upper_guardrail_success: float
    lower_guardrail_success: float
    upper_adjustment_fraction: float
    lower_adjustment_fraction: float
    adjustment_threshold: float
    adjustment_frequency: str
    spending_cap_option: str
    spending_floor_option: str
    final_value_target: float = 0.0
    cashflows: List[CashflowSetting] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Type coercion
        self.mode = str(self.mode)
        self.retirement_duration_months = int(self.retirement_duration_months)
        self.initial_value = float(self.initial_value)
        self.stock_pct = float(self.stock_pct)
        self.target_success_rate = float(self.target_success_rate)
        self.initial_monthly_spending = float(self.initial_monthly_spending)
        self.initial_spending_overridden = bool(self.initial_spending_overridden)
        self.upper_guardrail_success = float(self.upper_guardrail_success)
        self.lower_guardrail_success = float(self.lower_guardrail_success)
        self.upper_adjustment_fraction = float(self.upper_adjustment_fraction)
        self.lower_adjustment_fraction = float(self.lower_adjustment_fraction)
        self.adjustment_threshold = float(self.adjustment_threshold)
        self.adjustment_frequency = str(self.adjustment_frequency)
        self.spending_cap_option = str(self.spending_cap_option)
        self.spending_floor_option = str(self.spending_floor_option)
        self.final_value_target = float(self.final_value_target)

        # Validation
        if self.initial_value <= 0:
            raise ValueError(
                f"Initial portfolio value must be positive, got {self.initial_value:,.0f}"
            )
        if self.initial_monthly_spending < 0:
            raise ValueError(
                f"Initial monthly spending cannot be negative, got {self.initial_monthly_spending:,.0f}"
            )
        if self.retirement_duration_months <= 0:
            raise ValueError(
                f"Retirement duration must be positive, got {self.retirement_duration_months} months"
            )
        if not (0.0 <= self.stock_pct <= 1.0):
            raise ValueError(
                f"Stock percentage must be between 0% and 100%, got {self.stock_pct * 100:.0f}%"
            )
        if not (0.0 <= self.target_success_rate <= 1.0):
            raise ValueError(
                f"Target success rate must be between 0% and 100%, got {self.target_success_rate * 100:.0f}%"
            )
        if not (0.0 <= self.upper_guardrail_success <= 1.0):
            raise ValueError(
                f"Upper guardrail success rate must be between 0% and 100%, got {self.upper_guardrail_success * 100:.0f}%"
            )
        if not (0.0 <= self.lower_guardrail_success <= 1.0):
            raise ValueError(
                f"Lower guardrail success rate must be between 0% and 100%, got {self.lower_guardrail_success * 100:.0f}%"
            )
        if self.analysis_start_date > self.start_date:
            raise ValueError(
                f"Historical analysis start date ({self.analysis_start_date}) cannot be after "
                f"retirement start date ({self.start_date})"
            )
        if self.final_value_target < 0:
            raise ValueError(
                f"Final value target cannot be negative, got {self.final_value_target:,.0f}"
            )
        if self.final_value_target > self.initial_value:
            raise ValueError(
                f"Final value target ({self.final_value_target:,.0f}) cannot exceed "
                f"initial portfolio value ({self.initial_value:,.0f})"
            )

        # Clean cashflows
        cleaned: List[CashflowSetting] = []
        for flow in self.cashflows:
            if isinstance(flow, CashflowSetting):
                cleaned.append(flow)
            else:
                parsed = CashflowSetting.from_dict(flow)  # type: ignore[arg-type]
                if parsed is not None:
                    cleaned.append(parsed)
        self.cashflows = cleaned

    @property
    def spending_cap_multiplier(self) -> Optional[float]:
        return _relative_option_to_multiplier(self.spending_cap_option)

    @property
    def spending_floor_multiplier(self) -> Optional[float]:
        return _relative_option_to_multiplier(self.spending_floor_option)

    def cashflows_for_calculation(self) -> List[Dict[str, Any]]:
        return [flow.to_calculation_dict() for flow in self.cashflows]

    def simulation_signature(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "retirement_duration_months": int(self.retirement_duration_months),
            "analysis_start_date": self.analysis_start_date.isoformat() if self.analysis_start_date else None,
            "initial_value": float(self.initial_value),
            "stock_pct": float(self.stock_pct),
            "target_success_rate": float(self.target_success_rate),
            "upper_guardrail_success": float(self.upper_guardrail_success),
            "lower_guardrail_success": float(self.lower_guardrail_success),
            "upper_adjustment_fraction": float(self.upper_adjustment_fraction),
            "lower_adjustment_fraction": float(self.lower_adjustment_fraction),
            "adjustment_threshold": float(self.adjustment_threshold),
            "adjustment_frequency": self.adjustment_frequency,
            "spending_cap_multiplier": self.spending_cap_multiplier,
            "spending_floor_multiplier": self.spending_floor_multiplier,
            "cashflows": tuple(flow.signature() for flow in self.cashflows),
            "initial_monthly_spending": float(self.initial_monthly_spending),
            "initial_spending_overridden": bool(self.initial_spending_overridden),
            "final_value_target": float(self.final_value_target),
        }

    def retirement_end_date(self) -> Optional[dt.date]:
        if not self.start_date:
            return None
        months = int(self.retirement_duration_months)
        if months <= 0:
            return None
        start_ts = pd.to_datetime(self.start_date)
        result = start_ts + pd.DateOffset(months=months - 1)
        return result.date()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "retirement_duration_months": int(self.retirement_duration_months),
            "analysis_start_date": self.analysis_start_date.isoformat() if self.analysis_start_date else None,
            "initial_value": float(self.initial_value),
            "stock_pct": float(self.stock_pct),
            "target_success_rate": float(self.target_success_rate),
            "initial_monthly_spending": float(self.initial_monthly_spending),
            "initial_spending_overridden": bool(self.initial_spending_overridden),
            "upper_guardrail_success": float(self.upper_guardrail_success),
            "lower_guardrail_success": float(self.lower_guardrail_success),
            "upper_adjustment_fraction": float(self.upper_adjustment_fraction),
            "lower_adjustment_fraction": float(self.lower_adjustment_fraction),
            "adjustment_threshold": float(self.adjustment_threshold),
            "adjustment_frequency": self.adjustment_frequency,
            "spending_cap_option": self.spending_cap_option,
            "spending_floor_option": self.spending_floor_option,
            "final_value_target": float(self.final_value_target),
            "cashflows": [flow.to_serializable() for flow in self.cashflows],
        }

    def to_base64(self) -> str:
        payload = json.dumps(self.to_dict(), separators=(",", ":")).encode("utf-8")
        compressed = gzip.compress(payload)
        encoded = base64.urlsafe_b64encode(compressed).decode("utf-8")
        return encoded.rstrip("=")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Settings":
        def _parse_date(value: Any) -> dt.date:
            if isinstance(value, dt.date):
                return value
            if isinstance(value, str):
                return dt.date.fromisoformat(value)
            raise ValueError(f"Invalid date value: {value!r}")

        cashflows_data = [CashflowSetting.from_dict(item) for item in data.get("cashflows", [])]
        cashflows_clean = [cf for cf in cashflows_data if cf is not None]

        return cls(
            mode=data.get("mode", "Simulation Mode"),
            start_date=_parse_date(data.get("start_date", dt.date.today())),
            retirement_duration_months=data.get("retirement_duration_months", 360),
            analysis_start_date=_parse_date(data.get("analysis_start_date", dt.date(1871, 1, 1))),
            initial_value=data.get("initial_value", 1_000_000),
            stock_pct=data.get("stock_pct", 0.75),
            target_success_rate=data.get("target_success_rate", 0.9),
            initial_monthly_spending=data.get("initial_monthly_spending", 0.0),
            initial_spending_overridden=data.get("initial_spending_overridden", False),
            upper_guardrail_success=data.get("upper_guardrail_success", 1.0),
            lower_guardrail_success=data.get("lower_guardrail_success", 0.75),
            upper_adjustment_fraction=data.get("upper_adjustment_fraction", 1.0),
            lower_adjustment_fraction=data.get("lower_adjustment_fraction", 0.1),
            adjustment_threshold=data.get("adjustment_threshold", 0.05),
            adjustment_frequency=data.get("adjustment_frequency", "Monthly"),
            spending_cap_option=data.get("spending_cap_option", "Unlimited"),
            spending_floor_option=data.get("spending_floor_option", "Unlimited"),
            final_value_target=data.get("final_value_target", 0.0),
            cashflows=cashflows_clean,
        )

    @classmethod
    def from_base64(cls, payload: str) -> "Settings":
        padding = "=" * (-len(payload) % 4)
        try:
            decoded = base64.urlsafe_b64decode((payload + padding).encode("utf-8"))
        except Exception:
            raise ValueError(
                "The shared configuration link is corrupted or invalid. "
                "Please request a new link."
            )
        try:
            decompressed = gzip.decompress(decoded)
        except Exception:
            raise ValueError(
                "The shared configuration data could not be decompressed. "
                "The link may be incomplete or corrupted."
            )
        try:
            data = json.loads(decompressed.decode("utf-8"))
        except Exception:
            raise ValueError(
                "The shared configuration contains invalid data. "
                "Please request a new link."
            )
        if not isinstance(data, dict):
            raise ValueError(
                "The shared configuration format is invalid. "
                "Expected a configuration object."
            )
        return cls.from_dict(data)

    def apply_to_session_state(self, session_state: Any) -> None:
        session_state["app_mode"] = self.mode
        session_state["retirement_start_date"] = self.start_date
        session_state["retirement_duration_months"] = self.retirement_duration_months
        session_state["analysis_start_date"] = self.analysis_start_date
        session_state["initial_portfolio_value"] = self.initial_value
        session_state["stock_pct"] = self.stock_pct
        session_state["target_success_rate"] = self.target_success_rate
        session_state["initial_monthly_spending"] = self.initial_monthly_spending
        session_state["_initial_spending_overridden"] = bool(self.initial_spending_overridden)
        session_state["upper_guardrail_success"] = self.upper_guardrail_success
        session_state["lower_guardrail_success"] = self.lower_guardrail_success
        session_state["upper_adjustment_fraction"] = self.upper_adjustment_fraction
        session_state["lower_adjustment_fraction"] = self.lower_adjustment_fraction
        session_state["adjustment_threshold"] = self.adjustment_threshold
        session_state["adjustment_frequency"] = self.adjustment_frequency
        session_state["spending_cap_option"] = self.spending_cap_option
        session_state["spending_floor_option"] = self.spending_floor_option
        session_state["final_value_target"] = self.final_value_target
        session_state["cashflows"] = [flow.to_serializable() for flow in self.cashflows]

    def to_isr_params(self) -> Dict[str, Any]:
        """Return parameters for computing the initial spending rate."""
        return {
            "start_date": self.start_date,
            "duration_months": int(self.retirement_duration_months),
            "analysis_start_date": self.analysis_start_date,
            "initial_value": float(self.initial_value),
            "stock_pct": float(self.stock_pct),
            "desired_success_rate": float(self.target_success_rate),
            "final_value_target": float(self.final_value_target),
        }

    def to_guardrail_params(self) -> Dict[str, Any]:
        return {
            "start_date": self.start_date,
            "duration_months": int(self.retirement_duration_months),
            "analysis_start_date": self.analysis_start_date,
            "initial_value": float(self.initial_value),
            "stock_pct": float(self.stock_pct),
            "upper_sr": float(self.upper_guardrail_success),
            "lower_sr": float(self.lower_guardrail_success),
            "initial_spending": float(self.initial_monthly_spending),
            "final_value_target": float(self.final_value_target),
        }
