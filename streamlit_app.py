import calendar
import datetime

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional

import utils
import display
import controls

st.set_page_config(layout="wide", page_title="Guardrail Withdrawal Simulator")

# Mode toggle (top, persistent)
mode = st.radio(
    "Mode",
    ["Simulation Mode", "Guidance Mode"],
    index=0,
    horizontal=True,
    key="app_mode",
    label_visibility="collapsed"
)
is_guidance = (mode == "Guidance Mode")

title_ph = st.empty()
desc_ph = st.empty()

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

def _render_sidebar_label(text: str, color: Optional[str] = None) -> None:
    style = (
        "margin: 0 0 0.25rem 0;"
        " font-size: var(--font-size-sm, 0.875rem);"
        " font-weight: var(--font-weight-normal, 400);"
        " line-height: var(--line-height-sm, 1.4);"
    )
    if color:
        style += f" color: {color};"
    st.sidebar.markdown(f"<div style=\"{style}\">{text}</div>", unsafe_allow_html=True)

def _mark_initial_spending_overridden() -> None:
    """Flag that the current spending widget has been manually overridden."""
    st.session_state["_initial_spending_overridden"] = True

def _unmark_initial_spending_overridden() -> None:
    """Flag that the current spending widget is no longer being manually overridden."""
    st.session_state["_initial_spending_overridden"] = False

# Sidebar for inputs
st.sidebar.header("Simulation Parameters")

# Date Inputs

def _sidebar_month_year_selector(
    label: str,
    *,
    default_date: datetime.date,
    min_date: datetime.date,
    max_date: datetime.date,
    key_prefix: str,
    disabled: bool = False,
    help_text: Optional[str] = None,
) -> datetime.date:
    """Render year and month select boxes in the sidebar and return the first day of that month."""

    if default_date < min_date:
        default_date = min_date
    elif default_date > max_date:
        default_date = max_date

    year_key = f"{key_prefix}_year"
    month_key = f"{key_prefix}_month"
    saved_key = f"{key_prefix}_saved_selection"

    if disabled:
        previous_year = st.session_state.get(year_key)
        previous_month = st.session_state.get(month_key)
        if previous_year is not None and previous_month is not None:
            st.session_state[saved_key] = (previous_year, previous_month)
        st.session_state.pop(year_key, None)
        st.session_state.pop(month_key, None)
        restored_selection: Optional[tuple[int, int]] = None
    else:
        restored_selection = None
        if saved_key in st.session_state and (
            year_key not in st.session_state or month_key not in st.session_state
        ):
            restored_selection = st.session_state.pop(saved_key)

    year_options = list(range(min_date.year, max_date.year + 1))
    default_year = default_date.year

    container = st.sidebar.container()
    year_column, month_column = container.columns(2, gap="small")

    desired_year = default_year if disabled else st.session_state.get(year_key)
    if desired_year is None and restored_selection is not None:
        desired_year = restored_selection[0]
    if desired_year is None:
        desired_year = default_year
    if desired_year not in year_options:
        desired_year = max(min(desired_year, year_options[-1]), year_options[0])

    if not disabled:
        st.session_state[year_key] = desired_year

    with year_column:
        selected_year = year_column.selectbox(
            label,
            year_options,
            index=year_options.index(desired_year),
            key=year_key,
            disabled=disabled,
            help=help_text,
            on_change=_unmark_initial_spending_overridden,
        )

    month_start = min_date.month if selected_year == min_date.year else 1
    month_end = max_date.month if selected_year == max_date.year else 12
    month_options = list(range(month_start, month_end + 1))

    default_month = default_date.month
    if default_month not in month_options:
        default_month = month_options[0]

    desired_month = default_month if disabled else st.session_state.get(month_key)
    if desired_month is None and restored_selection is not None:
        desired_month = restored_selection[1]
    if desired_month is None:
        desired_month = default_month
    if desired_month not in month_options:
        desired_month = min(max(desired_month, month_options[0]), month_options[-1])

    if not disabled:
        st.session_state[month_key] = desired_month

    with month_column:
        selected_month = month_column.selectbox(
            "Month",
            month_options,
            index=month_options.index(desired_month),
            key=month_key,
            disabled=disabled,
            format_func=lambda m: calendar.month_name[m],
            on_change=_unmark_initial_spending_overridden,
        )

    return datetime.date(selected_year, selected_month, 1)


today_date = datetime.date.today()
start_date = _sidebar_month_year_selector(
    "Retirement Start Date",
    default_date=today_date if is_guidance else datetime.date(1968, 4, 1),
    min_date=datetime.date(1871, 1, 1),
    max_date=today_date,
    key_prefix="retirement_start",
    disabled=is_guidance,
    help_text="First retirement month used for withdrawals.\n\nIn Simulation Mode, the historical simulations begin from this date.\n\n"
              "In Guidance Mode, this defaults to today. Even if retirement is already underway, the guidance is forward looking from today.",
)

retirement_duration_months = st.sidebar.number_input(
    "Retirement Duration (months)",
    value=360,
    min_value=1,
    max_value=1200,
    step=12,
    on_change=_unmark_initial_spending_overridden,
    help="Length of retirement in months.\n\nIn Guidance Mode, this should be the remaining number of months, if retirement is already underway."
)

analysis_start_date = _sidebar_month_year_selector(
    "Historical Analysis Start Date",
    default_date=datetime.date(1871, 1, 1),
    min_date=datetime.date(1871, 1, 1),
    max_date=today_date,
    key_prefix="analysis_start",
    help_text="Earliest date for historical data included to estimate success rates (Shiller data begins in 1871).\n\nNote that when running historical "
             "simulations, each month's guardrails will be recalculated based on the historical data available between this start date and that month "
             "in history. A financial advisor running this strategy in the past would not have had a crystal ball to look into the future!",
)

# Numeric Inputs
initial_value = st.sidebar.number_input(
    "Initial Portfolio Value",
    value=1_000_000,
    min_value=100_000,
    step=100_000,
    on_change=_unmark_initial_spending_overridden,
    help="Starting portfolio balance in dollars at retirement."
)

stock_pct = st.sidebar.slider(
    "Stock Percentage",
    value=0.75,
    min_value=0.0,
    max_value=1.0,
    step=0.01,
    on_change=_unmark_initial_spending_overridden,
    help="Fraction of the portfolio allocated to US stocks; remainder to 10Y treasuries."
)

cap_options = ["Unlimited"] + [f"{pct}%" for pct in range(100, 201, 5)]
floor_options = ["Unlimited"] + [f"{pct}%" for pct in range(100, 24, -5)]

controls.sync_cashflows_from_widgets()

cashflows = controls.sanitize_cashflows(st.session_state.get("cashflows"))

# Compute Retirement End Date for Simulation Mode from duration
computed_end_date = (pd.to_datetime(start_date) + pd.DateOffset(months=int(retirement_duration_months) - 1)).date() if not is_guidance else None

# Compute Initial Withdrawal Rate (IWR) to show in the Target Success Rate label.
# We recompute only when relevant inputs change to avoid unnecessary calls.
iwr_params = {
    'start_date': pd.to_datetime(start_date),
    'duration_months': int(retirement_duration_months),
    'analysis_start_date': pd.to_datetime(analysis_start_date),
    'initial_value': float(initial_value),
    'stock_pct': float(stock_pct),
    # Use current target_success_rate if available, otherwise default of 0.90
    'desired_success_rate': float(st.session_state.get('target_success_rate', 0.90)),
    'cashflows': controls.cashflows_to_tuple(cashflows),
}
iwr_label_suffix = ""

try:
    if 'iwr_params' not in st.session_state or st.session_state['iwr_params'] != iwr_params:
        display.update_iwr_dynamic_label(iwr_params=iwr_params, is_guidance=is_guidance, cashflows=cashflows)
    iwr = st.session_state.get('iwr_value')
    if iwr is not None:
        iwr_label_suffix = f" (Initial WR: {iwr*100:.2f}%)"
        auto_initial_spending = round(float(initial_value) * float(iwr) / 12.0)
    else:
        iwr_label_suffix = " (Initial WR: N/A)"
        auto_initial_spending = None

except Exception as e:
    print(e)
    iwr_label_suffix = " (Initial WR: N/A)"
    auto_initial_spending = None

target_success_label = f"Target Success Rate{iwr_label_suffix}"
target_success_rate = st.sidebar.slider(
    target_success_label,
    value=st.session_state.get("target_success_rate", 0.90),
    min_value=0.0,
    max_value=1.0,
    step=0.01,
    help="Desired probability of success that will be used to select an initial withdrawal rate.\n\nThe initial "
         "withdrawal rate will be the rate at which fixed withdrawals over all periods of time with length = the "
         "configured retirement period length, between the Historical Analysis Start Date and the Retirement Start "
         "Date, end with >0 values this percent of the time.\n\nSetting this higher, e.g. 0.80-0.99, is more conservative: "
         "lower initial spending, lower chance of adjustment; setting this lower is more aggressive, 0.75-0.60 provides higher "
         "initial spending, higher chance of adjustment.",
    key="target_success_rate",
    on_change=_unmark_initial_spending_overridden,
)

# Current spending input is also an output when Target Success Rate moves, until it is manually overridden
if auto_initial_spending is not None:
    st.session_state["_initial_spending_auto_value"] = auto_initial_spending
    current_value = st.session_state.get("initial_monthly_spending")
    if st.session_state.get("_initial_spending_overridden") and current_value is not None and np.isclose(
        float(current_value), float(auto_initial_spending), rtol=0.0, atol=0.5
    ):
        st.session_state["_initial_spending_overridden"] = False
    if not st.session_state.get("_initial_spending_overridden"):
        if current_value is None or not np.isclose(float(current_value), float(auto_initial_spending), rtol=0.0, atol=0.5):
            st.session_state["initial_monthly_spending"] = float(auto_initial_spending)
else:
    st.session_state["_initial_spending_auto_value"] = None

if "initial_monthly_spending" not in st.session_state or st.session_state["initial_monthly_spending"] is None:
    st.session_state["initial_monthly_spending"] = 0.0

try:
    display.update_initial_spending_label(
        initial_spending=float(st.session_state.get("initial_monthly_spending", 0.0)),
        initial_value=float(initial_value),
        auto_spending=st.session_state.get("_initial_spending_auto_value"),
        overridden=bool(st.session_state.get("_initial_spending_overridden", False)),
    )
except Exception as e:
    print(e)
    st.session_state['initial_spending_label_text'] = "Initial Monthly Spending (WR: N/A)"
    st.session_state['initial_spending_label_color'] = None

initial_spending_label = st.session_state.get('initial_spending_label_text', "Initial Monthly Spending (WR: N/A)")
initial_spending_color = st.session_state.get('initial_spending_label_color')

_render_sidebar_label(initial_spending_label, initial_spending_color)

initial_monthly_spending = st.sidebar.number_input(
    initial_spending_label,
    min_value=0.0,
    step=10.0,
    format="%.0f",
    help="The initial monthly spending level for the retirement simulation, which will be used until a guardrail is hit.\n\n"
         "Automatically updates as you change the Target Withdrawal Rate, but can be set to a custom value if desired.\n\n",
    key="initial_monthly_spending",
    on_change=_mark_initial_spending_overridden,
    label_visibility="collapsed",
)

# Compute dynamic labels for Guardrail Success Rates showing initial (first period) PVs
upper_label_suffix = ""
lower_label_suffix = ""
try:
    gr_params = {
        'start_date': pd.to_datetime(start_date),
        'duration_months': int(retirement_duration_months),
        'analysis_start_date': pd.to_datetime(analysis_start_date),
        'initial_value': float(initial_value),
        'stock_pct': float(stock_pct),
        'upper_sr': float(st.session_state.get("upper_guardrail_success", 1.00)),
        'lower_sr': float(st.session_state.get("lower_guardrail_success", 0.75)),
        'iwr': float(st.session_state.get('iwr_value')) if st.session_state.get('iwr_value') is not None else None,
        'initial_spending': float(st.session_state.get("initial_monthly_spending", 0.0)),
        'cashflows': controls.cashflows_to_tuple(cashflows),
    }

    if ('guardrail_params' not in st.session_state) or (st.session_state['guardrail_params'] != gr_params):
        display.update_guardrail_dynamic_labels(gr_params=gr_params, is_guidance=is_guidance, cashflows=cashflows)

    upper_label_suffix = st.session_state.get('upper_label_suffix', " (Initial PV: N/A)")
    lower_label_suffix = st.session_state.get('lower_label_suffix', " (Initial PV: N/A)")

except Exception as e:
    print(e)
    upper_label_suffix = " (Initial PV: N/A)"
    lower_label_suffix = " (Initial PV: N/A)"
    st.session_state['upper_label_color'] = None
    st.session_state['lower_label_color'] = None

upper_guardrail_label = f"Upper Guardrail Success Rate{upper_label_suffix}"
lower_guardrail_label = f"Lower Guardrail Success Rate{lower_label_suffix}"

upper_label_color = st.session_state.get('upper_label_color')
lower_label_color = st.session_state.get('lower_label_color')

_render_sidebar_label(upper_guardrail_label, upper_label_color)

upper_guardrail_success = st.sidebar.slider(
    upper_guardrail_label,
    value=st.session_state.get("upper_guardrail_success", 1.00),
    min_value=0.0,
    max_value=1.0,
    step=0.01,
    help="The withdrawal rate used to calculate the upper guardrail portfolio value.\n\nThis is the value where the "
         "current withdrawal amount, if held constant, will succeed this frequently or more, for all periods with "
         "length = # months remaining in retirement, between the Historical Analysis Start Date and the current "
         "simulation date.\n\nSetting this higher is more conservative, and will cause you to wait longer to increase "
         "your spending when markets are up.",
    key="upper_guardrail_success",
    label_visibility="collapsed"
)

_render_sidebar_label(lower_guardrail_label, lower_label_color)

lower_guardrail_success = st.sidebar.slider(
    lower_guardrail_label,
    value=st.session_state.get("lower_guardrail_success", 0.75),
    min_value=0.0,
    max_value=1.0,
    step=0.01,
    help="The withdrawal rate used to calculate the lower guardrail portfolio value.\n\nThis is the value where the "
         "current withdrawal amount, if held constant, will succeed this frequently or less, for all periods with "
         "length = # months remaining in retirement, between the Historical Analysis Start Date and the current "
         "simulation date.\n\nSetting this higher is more conservative, and will cause you to decrease your spending "
         "sooner when markets are down.",
    key="lower_guardrail_success",
    label_visibility="collapsed"
)

upper_adjustment_fraction = st.sidebar.slider(
    "Upper Adjustment Fraction",
    value=1.0,
    min_value=0.0,
    max_value=1.0,
    step=0.05,
    help="How much to increase spending when we hit the upper guardrail.\n\nExpressed as a % of the distance between "
         "the Upper Guardrail Success Rate and the Target Success Rate.\n\nFor example, if the upper guardrail "
         "represents 100% success and the target is 90%, setting this value to 50% means we go half the distance back "
         "to the target, and our new withdrawal rate will be based on a 95% chance of success.\n\nSetting this higher "
         "is more aggressive, and will cause you to make larger spending increases when you hit the upper guardrail."
)

lower_adjustment_fraction = st.sidebar.slider(
    "Lower Adjustment Fraction",
    value=0.1,
    min_value=0.0,
    max_value=1.0,
    step=0.05,
    help="How much to decrease spending when we hit the lower guardrail.\n\nExpressed as a % of the distance between "
         "the Lower Guardrail Success Rate and the Target Success Rate.\n\nFor example, if the lower guardrail "
         "represents 70% success and the target is 90%, setting this value to 50% means we go half the distance back "
         "to the target, and our new withdrawal rate will be based on an 80% chance of success.\n\nSetting this higher "
         "is more conservative, and will cause you to make larger spending decreases when you hit the lower guardrail."
)

adjustment_threshold = st.sidebar.slider(
    "Adjustment Threshold (e.g., 0.05 for 5%)",
    value=0.0 if is_guidance else 0.05,
    min_value=0.0,
    max_value=0.2,
    step=0.01,
    help="The minimum percent difference between our new spending and our prior spending, before we make a change.\n\n"
         "Even if we hit a guardrail, we may elect to set this to 5% to avoid making lots of small adjustments. Set it "
         "to 0% to disable it and allow all guardrail hits to trigger spending adjustments.\n\nSetting this higher is "
         "neither aggressive nor conservative, since it impacts spending increases as well as decreases. It is purely "
         "a question of whether frequent adjustments are acceptable and administratively feasible for the client.\n\n"
         "Not used in Guidance Mode, which will always show spending adjustment suggestions, no matter how small.",
    disabled=is_guidance  # In Guidance Mode, single-run snapshot ignores threshold
)

adjustment_frequency = st.sidebar.selectbox(
    "Adjustment Frequency",
    options=["Monthly", "Quarterly", "Biannually", "Annually"],
    index=0,
    help="How often spending adjustments are permitted. Choosing Quarterly, Biannually, or Annually restricts guardrail checks "
         "and any resulting spending changes to the beginning of those periods (Jan/Apr/Jul/Oct, Jan/Jul, or January).",
    disabled=is_guidance  # In Guidance Mode, no decision gating, it's up to the adviser and client
)

def _relative_option_to_multiplier(option: str):
    if option == "Unlimited":
        return None
    try:
        return float(option.strip("%")) / 100.0
    except (TypeError, ValueError):
        return None

spending_cap_multiplier = _relative_option_to_multiplier(st.session_state.get("spending_cap_option"))
spending_floor_multiplier = _relative_option_to_multiplier(st.session_state.get("spending_floor_option"))

# Build current simulation parameters dict for change detection
sim_params = {
    'start_date': pd.to_datetime(start_date),
    'duration_months': int(retirement_duration_months),
    'analysis_start_date': pd.to_datetime(analysis_start_date),
    'initial_value': float(initial_value),
    'stock_pct': float(stock_pct),
    'target_success_rate': float(target_success_rate),
    'upper_guardrail_success': float(upper_guardrail_success),
    'lower_guardrail_success': float(lower_guardrail_success),
    'upper_adjustment_fraction': float(upper_adjustment_fraction),
    'lower_adjustment_fraction': float(lower_adjustment_fraction),
    'adjustment_threshold': float(adjustment_threshold),
    'adjustment_frequency': adjustment_frequency,
    'spending_cap_multiplier': spending_cap_multiplier,
    'spending_floor_multiplier': spending_floor_multiplier,
    'cashflows': controls.cashflows_to_tuple(cashflows),
}
last_run_params = st.session_state.get('last_run_params')
dirty = last_run_params is not None and last_run_params != sim_params
st.session_state['dirty'] = dirty

DIRTY_COLOR = "#8B0000"  # dark red

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

with st.sidebar.expander("Advanced Controls"):
    st.selectbox(
        "Spending Cap",
        options=cap_options,
        key="spending_cap_option",
        help="Maximum spending level as a percent of the initial monthly spending.",
        disabled=is_guidance  # In Guidance Mode, no decision gating, it's up to the adviser and client
    )
    st.selectbox(
        "Spending Floor",
        options=floor_options,
        key="spending_floor_option",
        help="Minimum spending level as a percent of the initial monthly spending.",
        disabled = is_guidance  # In Guidance Mode, no decision gating, it's up to the adviser and client
    )

    if st.button("Add Recurring Cashflow", key="add_cashflow_btn"):
        st.session_state["cashflows"].append({
            "start_month": 0,
            "end_month": 0,
            "amount": 0.0,
            "label": f"Cashflow {len(st.session_state['cashflows']) + 1}",
        })
        st.rerun()

    for idx, flow in enumerate(st.session_state["cashflows"]):
        controls.ensure_cashflow_widget_state(idx, flow)

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
            controls.clear_cashflow_widget_state(idx)
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


# When inputs change, visually dim and surround the main area with a red border
if dirty and not is_guidance:
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


if is_guidance:
    # Guidance Mode: run a single-iteration snapshot and display text output

    # Load Shiller data (cached)
    shiller_df = st.session_state.get('shiller_df')
    if shiller_df is None:
        shiller_df = utils.load_shiller_data()
        st.session_state['shiller_df'] = shiller_df

    # Use the latest available Shiller data date (<= today) as the as-of date.
    today = datetime.date.today()
    latest_shiller_date = pd.to_datetime(shiller_df["Date"].max()).date()
    asof_date = latest_shiller_date if latest_shiller_date <= today else today

    try:
        snap = utils.compute_guardrail_guidance_snapshot(
            df=shiller_df,
            asof_date=asof_date,
            duration_months=int(retirement_duration_months),
            analysis_start_date=analysis_start_date,
            current_portfolio_value=initial_value,
            initial_monthly_spending=initial_monthly_spending,
            stock_pct=stock_pct,
            target_success_rate=target_success_rate,
            upper_guardrail_success=upper_guardrail_success,
            lower_guardrail_success=lower_guardrail_success,
            upper_adjustment_fraction=upper_adjustment_fraction,
            lower_adjustment_fraction=lower_adjustment_fraction,
            cashflows=cashflows
        )

        # TODO for https://github.com/rogercost/fire-guardrails/issues/13
        # Create a nice tabular or graphic display, move logic to display module

        def fmt_money(x):
            # Escape the dollar sign so Streamlit Markdown doesn't interpret $...$ as LaTeX math
            return f"\\${x:,.0f}" if (x is not None and (isinstance(x, (int, float)) and not np.isinf(x))) else "N/A"

        def fmt_pct(p):
            return f"{p*100:+.0f}%" if p is not None else "N/A"

        start_wr = st.session_state.get("iwr_value", snap.get("target_withdrawal_rate"))
        start_month = snap.get("target_monthly_spending")
        start_year = (start_month * 12.0) if start_month is not None else None
        start_net_withdrawal = snap.get("target_monthly_withdrawal")

        upper_val = snap.get("upper_guardrail_value")
        lower_val = snap.get("lower_guardrail_value")

        up_adj_pct = snap.get("upper_adjustment_pct")
        up_adj_month = snap.get("upper_adjusted_monthly")
        up_adj_year = (up_adj_month * 12.0) if up_adj_month is not None else None

        low_adj_pct = snap.get("lower_adjustment_pct")
        low_adj_month = snap.get("lower_adjusted_monthly")
        low_adj_year = (low_adj_month * 12.0) if low_adj_month is not None else None

        cashflow_month0 = snap.get("current_cashflow")

        def fmt_month(ts):
            if ts is None or pd.isna(ts):
                return "N/A"
            timestamp = pd.to_datetime(ts)
            return timestamp.strftime("%B %Y")

        st.subheader("Guidance Mode")
        st.markdown("Use this mode to generate forward-looking guidance for a client who is retired today and in drawdown.\n\n"
                    "For more information, see the [official documentation](https://github.com/rogercost/fire-guardrails/blob/main/README.md).")

        if start_wr is not None:
            st.markdown(
                f"* **Target Withdrawal Rate:** {start_wr*100:.2f}% "
                f"({fmt_money(start_month)}/month or {fmt_money(start_year)}/year total spending based on the Initial Portfolio Value) "
                f"— portfolio withdrawal after cashflows: {fmt_money(start_net_withdrawal)}/month"
            )
        else:
            st.markdown("**Starting Withdrawal Rate:** N/A")

        st.markdown(f"* **Month 1 Cashflows:** {fmt_money(cashflow_month0)}/month")

        st.markdown(
            f"* **Upper Guardrail Portfolio Value:** {fmt_money(upper_val)} based on the Current Monthly Spending\n  * If client's portfolio value exceeds this, adjust "
            f"spending by {fmt_pct(up_adj_pct)} to {fmt_money(up_adj_month)}/month or {fmt_money(up_adj_year)}/year"
        )
        st.markdown(
            f"* **Lower Guardrail Portfolio Value:** {fmt_money(lower_val)} based on the Current Monthly Spending\n  * If client's portfolio value falls below this, adjust "
            f"spending by {fmt_pct(low_adj_pct)} to {fmt_money(low_adj_month)}/month or {fmt_money(low_adj_year)}/year"
        )

    except Exception as e:
        st.error(f"Unable to compute guidance snapshot: {e}")

elif st.sidebar.button(
    "Run Simulation",
    help="Fetch data and run the guardrail withdrawal simulation with the selected parameters."
):
    status_ph = st.empty()
    status_ph.text("Loading Shiller data...")
    shiller_df = st.session_state.get('shiller_df')
    if shiller_df is None:
        shiller_df = utils.load_shiller_data()
        st.session_state['shiller_df'] = shiller_df
    status_ph.text("Shiller data loaded.")

    # Progress bar shown in the same status line area (0% -> 100%)
    progress = status_ph.progress(0, text="Calculating guardrail withdrawals... 0%")
    state = {"pct": 0, "status": None}

    def render_progress():
        label = f"Calculating guardrail withdrawals... {state['pct']}%"
        if state["status"]:
            label = f"{label} — {state['status']}"
        progress.progress(state["pct"], text=label)

    def on_progress(current, total):
        pct = int(current * 100 / total) if total else 0
        state["pct"] = pct
        render_progress()

    def on_status(msg):
        state["status"] = msg
        render_progress()

    results_df = utils.get_guardrail_withdrawals(
        df=shiller_df,
        start_date=start_date,
        end_date=computed_end_date,
        analysis_start_date=analysis_start_date,
        initial_value=initial_value,
        initial_monthly_spending=initial_monthly_spending,
        stock_pct=stock_pct,
        target_success_rate=target_success_rate,
        upper_guardrail_success=upper_guardrail_success,
        lower_guardrail_success=lower_guardrail_success,
        upper_adjustment_fraction=upper_adjustment_fraction,
        lower_adjustment_fraction=lower_adjustment_fraction,
        adjustment_threshold=adjustment_threshold,
        adjustment_frequency=adjustment_frequency,
        spending_cap=spending_cap_multiplier,
        spending_floor=spending_floor_multiplier,
        cashflows=cashflows,
        verbose=True,
        on_progress=on_progress,
        on_status=on_status
    )

    # Cache results in session state for re-render without recomputation
    st.session_state['results_df'] = results_df
    st.session_state['last_run_params'] = sim_params
    st.session_state['dirty'] = False

    # Remove status line entirely to place plots at the very top
    status_ph.empty()

    display.render_simulation_results(results_df)

elif not is_guidance:
    if 'results_df' in st.session_state:
        results_df = st.session_state['results_df']

        if st.session_state.get('dirty'):
            render_dirty_banner()

        display.render_simulation_results(results_df)
    else:
        st.subheader("Simulation Mode")
        st.markdown("Use this mode to simulate running a guardrail-based retirement withdrawal strategy during a historical period.\n\n"
                    "For more information, see the [documentation](https://github.com/rogercost/fire-guardrails/blob/main/README.md).")
        st.info("Adjust parameters in the sidebar and click 'Run Simulation' to start.")
