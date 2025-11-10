import streamlit as st


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
