import datetime

import pandas as pd
import streamlit as st

from app_settings import Settings

DIRTY_COLOR = "#8B0000"  # dark red

# Helpers for cashflow management
def sanitize_cashflows(raw_cashflows):
    """Convert raw cashflow inputs into validated dictionaries with numeric values."""

    sanitized = []
    for flow in raw_cashflows or []:
        try:
            start = int(flow.get("start_month", 0))
            end = int(flow.get("end_month", 0))
            amount = float(flow.get("amount", 0.0))
        except (AttributeError, TypeError, ValueError):
            continue

        if end < start:
            continue

        label_raw = flow.get("label") if isinstance(flow, dict) else None
        label = str(label_raw).strip() if label_raw is not None else None

        sanitized.append({
            "start_month": start,
            "end_month": end,
            "amount": amount,
            "label": label,
        })
    return sanitized


def cashflows_to_tuple(cashflows):
    """Create a hashable snapshot of each cashflow's timing and amount."""

    return tuple((cf["start_month"], cf["end_month"], cf["amount"]) for cf in cashflows)


def clear_cashflow_widget_state(start_idx: int = 0) -> None:
    """Remove cached widget values for cashflow inputs from the specified starting index onward."""

    prefixes = (
        "cf_start_",
        "cf_end_",
        "cf_amount_",
        "cf_label_",
    )

    keys_to_drop = []
    for key in list(st.session_state.keys()):
        for prefix in prefixes:
            if key.startswith(prefix):
                suffix = key[len(prefix) :]
                if suffix.isdigit() and int(suffix) >= start_idx:
                    keys_to_drop.append(key)
                break

    for key in keys_to_drop:
        del st.session_state[key]


# Ensure widget defaults for existing cashflows remain synchronized prior to rendering inputs
def ensure_cashflow_widget_state(idx: int, flow: dict) -> None:
    """Keep a cashflow row's widget state aligned with sanitized default values."""

    start_key = f"cf_start_{idx}"
    end_key = f"cf_end_{idx}"
    amount_key = f"cf_amount_{idx}"
    label_key = f"cf_label_{idx}"

    default_start = int(flow.get("start_month", 0))
    default_end = int(flow.get("end_month", default_start))
    default_amount = float(flow.get("amount", 0.0))
    default_label = str(flow.get("label") or f"Cashflow {idx + 1}")

    if start_key not in st.session_state:
        st.session_state[start_key] = default_start

    if end_key not in st.session_state:
        st.session_state[end_key] = max(default_end, st.session_state[start_key])
    elif st.session_state[end_key] < st.session_state[start_key]:
        st.session_state[end_key] = st.session_state[start_key]

    if amount_key not in st.session_state:
        st.session_state[amount_key] = default_amount

    if label_key not in st.session_state or not str(st.session_state[label_key]).strip():
        st.session_state[label_key] = default_label


def sync_cashflows_from_widgets() -> None:
    """Write the current widget state back into the tracked cashflow configurations."""

    flows = st.session_state.get("cashflows")
    if not flows:
        return

    for idx, flow in enumerate(flows):
        start_key = f"cf_start_{idx}"
        end_key = f"cf_end_{idx}"
        amount_key = f"cf_amount_{idx}"
        label_key = f"cf_label_{idx}"

        try:
            start_val = int(st.session_state.get(start_key, flow.get("start_month", 0)))
        except (TypeError, ValueError):
            start_val = int(flow.get("start_month", 0))

        try:
            end_val = int(st.session_state.get(end_key, flow.get("end_month", start_val)))
        except (TypeError, ValueError):
            end_val = int(flow.get("end_month", start_val))

        if end_val < start_val:
            end_val = start_val

        try:
            amount_val = float(st.session_state.get(amount_key, flow.get("amount", 0.0)))
        except (TypeError, ValueError):
            amount_val = float(flow.get("amount", 0.0))

        label_raw = st.session_state.get(label_key, flow.get("label") or f"Cashflow {idx + 1}")
        label_val = str(label_raw).strip() or f"Cashflow {idx + 1}"

        st.session_state[start_key] = start_val
        st.session_state[end_key] = end_val
        st.session_state[amount_key] = amount_val
        st.session_state[label_key] = label_val

        flows[idx] = {
            "start_month": start_val,
            "end_month": end_val,
            "amount": amount_val,
            "label": label_val,
        }


def get_date_state(key: str, default: datetime.date) -> datetime.date:
    """Return a stored date value, falling back to the provided baseline when parsing fails."""

    value = st.session_state.get(key)
    if value is None:
        return default
    if isinstance(value, datetime.date):
        return value
    try:
        return pd.to_datetime(value).date()
    except Exception:
        return default


def get_int_state(key: str, default: int) -> int:
    """Return an integer from session state, using the fallback when conversion is unsafe."""

    value = st.session_state.get(key)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_float_state(key: str, default: float) -> float:
    """Return a floating-point number from session state, with graceful fallback on errors."""

    value = st.session_state.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def init_start_date_field(today_date, sim_start_key, start_date_key, start_date_default, mode, is_guidance):
    """Manage the retirement start date widget when switching between simulation and guidance modes."""

    previous_mode_key = "_previous_mode"

    if sim_start_key not in st.session_state:
        st.session_state[sim_start_key] = get_date_state(start_date_key, start_date_default)

    previous_mode = st.session_state.get(previous_mode_key)
    if previous_mode != mode:
        if is_guidance:
            st.session_state[sim_start_key] = get_date_state(start_date_key, start_date_default)
            st.session_state[start_date_key] = today_date
        else:
            restored_start = st.session_state.get(sim_start_key, start_date_default)
            if not isinstance(restored_start, datetime.date):
                restored_start = get_date_state(start_date_key, start_date_default)
            st.session_state[start_date_key] = restored_start
        st.session_state[previous_mode_key] = mode

    if start_date_key not in st.session_state:
        default_start = today_date if is_guidance else st.session_state.get(sim_start_key, start_date_default)
        if not isinstance(default_start, datetime.date):
            default_start = start_date_default
        st.session_state[start_date_key] = default_start


def initialize_display():
    """Populate session state with default values required for the control widgets."""

    if "show_advanced_modal" not in st.session_state:
        st.session_state["show_advanced_modal"] = False
    if "spending_cap_option" not in st.session_state:
        st.session_state["spending_cap_option"] = "Unlimited"
    if "spending_floor_option" not in st.session_state:
        st.session_state["spending_floor_option"] = "Unlimited"
    if "cashflows" not in st.session_state:
        st.session_state["cashflows"] = []
    if "conditional_cashflows" not in st.session_state:
        st.session_state["conditional_cashflows"] = []
    if "_initial_spending_overridden" not in st.session_state:
        st.session_state["_initial_spending_overridden"] = False
    if "_initial_spending_auto_value" not in st.session_state:
        st.session_state["_initial_spending_auto_value"] = None
    if "final_value_target" not in st.session_state:
        st.session_state["final_value_target"] = 0.0


def draw_cashflow_widget_rows():
    """Render editable cashflow rows and keep their widgets synchronized."""

    for idx, flow in enumerate(st.session_state["cashflows"]):
        ensure_cashflow_widget_state(idx, flow)

        name_col, remove_col = st.columns([1, 0.15])
        label_key = f"cf_label_{idx}"
        name_col.text_input(
            "Cashflow Name",
            key=label_key,
            label_visibility="collapsed",
            placeholder="Cashflow name",
        )
        if remove_col.button("✕", key=f"cf_remove_{idx}"):
            st.session_state["cashflows"].pop(idx)
            clear_cashflow_widget_state(idx)
            st.rerun()

        col_start, col_end, col_amount = st.columns(3)
        col_start.number_input(
            "Start Month",
            min_value=0,
            step=1,
            key=f"cf_start_{idx}",
        )
        col_end.number_input(
            "End Month",
            min_value=0,
            step=1,
            key=f"cf_end_{idx}",
        )
        col_amount.number_input(
            "Amount ($/mo)",
            step=50.0,
            format="%0.0f",
            key=f"cf_amount_{idx}",
        )


# Helpers for conditional cashflow management
CONDITIONAL_THRESHOLD_OPTIONS = tuple(f"{pct}%" for pct in range(5, 100, 5))
CONDITIONAL_TYPE_OPTIONS = ("One-time", "Recurring")


def sanitize_conditional_cashflows(raw_cashflows):
    """Convert raw conditional cashflow inputs into validated dictionaries."""

    sanitized = []
    for flow in raw_cashflows or []:
        try:
            cashflow_type = str(flow.get("cashflow_type", "one_time"))
            if cashflow_type not in ("one_time", "recurring"):
                continue
            trigger_threshold = str(flow.get("trigger_threshold", "50%"))
            amount = float(flow.get("amount", 0.0))
        except (AttributeError, TypeError, ValueError):
            continue

        label_raw = flow.get("label") if isinstance(flow, dict) else None
        label = str(label_raw).strip() if label_raw is not None else None

        sanitized.append({
            "cashflow_type": cashflow_type,
            "trigger_threshold": trigger_threshold,
            "amount": amount,
            "label": label,
        })
    return sanitized


def clear_conditional_cashflow_widget_state(start_idx: int = 0) -> None:
    """Remove cached widget values for conditional cashflow inputs from the specified index onward."""

    prefixes = (
        "ccf_type_",
        "ccf_threshold_",
        "ccf_amount_",
        "ccf_label_",
    )

    keys_to_drop = []
    for key in list(st.session_state.keys()):
        for prefix in prefixes:
            if key.startswith(prefix):
                suffix = key[len(prefix):]
                if suffix.isdigit() and int(suffix) >= start_idx:
                    keys_to_drop.append(key)
                break

    for key in keys_to_drop:
        del st.session_state[key]


def ensure_conditional_cashflow_widget_state(idx: int, flow: dict) -> None:
    """Keep a conditional cashflow row's widget state aligned with sanitized default values."""

    type_key = f"ccf_type_{idx}"
    threshold_key = f"ccf_threshold_{idx}"
    amount_key = f"ccf_amount_{idx}"
    label_key = f"ccf_label_{idx}"

    default_type = str(flow.get("cashflow_type", "one_time"))
    # Convert internal value to display value
    default_type_display = "One-time" if default_type == "one_time" else "Recurring"
    default_threshold = str(flow.get("trigger_threshold", "50%"))
    default_amount = float(flow.get("amount", 0.0))
    default_label = str(flow.get("label") or f"Conditional {idx + 1}")

    if type_key not in st.session_state:
        st.session_state[type_key] = default_type_display

    if threshold_key not in st.session_state:
        st.session_state[threshold_key] = default_threshold

    if amount_key not in st.session_state:
        st.session_state[amount_key] = default_amount

    if label_key not in st.session_state or not str(st.session_state[label_key]).strip():
        st.session_state[label_key] = default_label


def sync_conditional_cashflows_from_widgets() -> None:
    """Write the current widget state back into the tracked conditional cashflow configurations."""

    flows = st.session_state.get("conditional_cashflows")
    if not flows:
        return

    for idx, flow in enumerate(flows):
        type_key = f"ccf_type_{idx}"
        threshold_key = f"ccf_threshold_{idx}"
        amount_key = f"ccf_amount_{idx}"
        label_key = f"ccf_label_{idx}"

        type_display = st.session_state.get(type_key, "One-time")
        # Convert display value to internal value
        type_val = "one_time" if type_display == "One-time" else "recurring"

        threshold_val = st.session_state.get(threshold_key, flow.get("trigger_threshold", "50%"))

        try:
            amount_val = float(st.session_state.get(amount_key, flow.get("amount", 0.0)))
        except (TypeError, ValueError):
            amount_val = float(flow.get("amount", 0.0))

        label_raw = st.session_state.get(label_key, flow.get("label") or f"Conditional {idx + 1}")
        label_val = str(label_raw).strip() or f"Conditional {idx + 1}"

        st.session_state[type_key] = type_display
        st.session_state[threshold_key] = threshold_val
        st.session_state[amount_key] = amount_val
        st.session_state[label_key] = label_val

        flows[idx] = {
            "cashflow_type": type_val,
            "trigger_threshold": threshold_val,
            "amount": amount_val,
            "label": label_val,
        }


def draw_conditional_cashflow_widget_rows():
    """Render editable conditional cashflow rows and keep their widgets synchronized."""

    for idx, flow in enumerate(st.session_state.get("conditional_cashflows", [])):
        ensure_conditional_cashflow_widget_state(idx, flow)

        # Row 1: Name and remove button
        name_col, remove_col = st.columns([1, 0.15])
        label_key = f"ccf_label_{idx}"
        name_col.text_input(
            "Conditional Cashflow Name",
            key=label_key,
            label_visibility="collapsed",
            placeholder="Conditional cashflow name",
        )
        if remove_col.button("✕", key=f"ccf_remove_{idx}"):
            st.session_state["conditional_cashflows"].pop(idx)
            clear_conditional_cashflow_widget_state(idx)
            st.rerun()

        # Row 2: Type, Threshold, Amount
        col_type, col_threshold, col_amount = st.columns(3)
        col_type.selectbox(
            "Type",
            options=CONDITIONAL_TYPE_OPTIONS,
            key=f"ccf_type_{idx}",
        )
        col_threshold.selectbox(
            "Trigger Below",
            options=CONDITIONAL_THRESHOLD_OPTIONS,
            key=f"ccf_threshold_{idx}",
            help="Cashflow triggers when spending falls below this % of initial spending",
        )
        col_amount.number_input(
            "Amount ($/mo)",
            step=50.0,
            format="%0.0f",
            key=f"ccf_amount_{idx}",
        )


def render_dirty_banner():
    """Display a warning banner indicating that the inputs changed and a rerun is needed."""

    st.markdown(
        f"""
        <div style="
            padding: 0.75rem 1rem;
            margin: 0 0 0.75rem 0;
            border: 1px solid {DIRTY_COLOR};
            background: rgba(139,0,0,0.08);
            color: {DIRTY_COLOR};
            border-radius: 6px;
            font-weight: 600;">
            Inputs changed, please rerun
        </div>
        """,
        unsafe_allow_html=True
    )

def draw_dirty_border():
    """Add a temporary border effect around the app to reinforce the rerun prompt."""

    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] > .main {{
            border: 3px solid {DIRTY_COLOR};
            border-radius: 8px;
            padding: 6px;
            filter: grayscale(30%) brightness(0.95);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def hydrate_settings():
    """Load shared configuration details from the URL and apply them to the session."""

    query_params = st.query_params
    config_value = query_params.get("config") if query_params is not None else None
    if config_value:
        try:
            loaded_settings = Settings.from_base64(config_value)
            loaded_settings.apply_to_session_state(st.session_state)
            st.session_state["settings"] = loaded_settings
            st.session_state["_settings_loaded_from_query"] = True
        except Exception as exc:
            st.session_state["_settings_error"] = str(exc)
    st.session_state["_settings_initialized"] = True