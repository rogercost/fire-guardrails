import datetime

import pandas as pd
import streamlit as st

from app_settings import Settings

DIRTY_COLOR = "#8B0000"  # dark red

# Helpers for cashflow management
def sanitize_cashflows(raw_cashflows):
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
    return tuple((cf["start_month"], cf["end_month"], cf["amount"]) for cf in cashflows)


def clear_cashflow_widget_state(start_idx: int = 0) -> None:
    """Remove cached widget values for cashflows starting from ``start_idx``."""

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
    value = st.session_state.get(key)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_float_state(key: str, default: float) -> float:
    value = st.session_state.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def init_start_date_field(today_date, sim_start_key, start_date_key, start_date_default, mode, is_guidance):
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
    if "show_advanced_modal" not in st.session_state:
        st.session_state["show_advanced_modal"] = False
    if "spending_cap_option" not in st.session_state:
        st.session_state["spending_cap_option"] = "Unlimited"
    if "spending_floor_option" not in st.session_state:
        st.session_state["spending_floor_option"] = "Unlimited"
    if "cashflows" not in st.session_state:
        st.session_state["cashflows"] = []
    if "_initial_spending_overridden" not in st.session_state:
        st.session_state["_initial_spending_overridden"] = False
    if "_initial_spending_auto_value" not in st.session_state:
        st.session_state["_initial_spending_auto_value"] = None


def draw_cashflow_widget_rows():
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

def render_dirty_banner():
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