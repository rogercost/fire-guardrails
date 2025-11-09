import datetime

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

import utils

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
if "_current_spending_overridden" not in st.session_state:
    st.session_state["_current_spending_overridden"] = False
if "_current_spending_auto_value" not in st.session_state:
    st.session_state["_current_spending_auto_value"] = None


# Helpers for cashflow management
def _sanitize_cashflows(raw_cashflows):
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

        sanitized.append({
            "start_month": start,
            "end_month": end,
            "amount": amount,
        })
    return sanitized


def _cashflows_to_tuple(cashflows):
    return tuple((cf["start_month"], cf["end_month"], cf["amount"]) for cf in cashflows)


def _clear_cashflow_widget_state(start_idx: int = 0) -> None:
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


def _mark_current_spending_overridden() -> None:
    """Flag that the current spending widget has been manually overridden."""

    st.session_state["_current_spending_overridden"] = True


# Sidebar for inputs
st.sidebar.header("Simulation Parameters")

# Date Inputs
today_date = datetime.date.today()
start_date = st.sidebar.date_input(
    "Retirement Start Date",
    value=today_date if is_guidance else datetime.date(1968, 4, 1),
    min_value=datetime.date(1871, 1, 1),
    max_value=today_date,
    help="First retirement month used for withdrawals.\n\nIn Simulation Mode, the historical simulations begin from this date.\n\n"
         "In Guidance Mode, this defaults to today. Even if retirement is already underway, the guidance is forward looking from today.",
    disabled=is_guidance  # In Guidance Mode, this is fixed to today
)
retirement_duration_months = st.sidebar.number_input(
    "Retirement Duration (months)",
    value=360,
    min_value=1,
    max_value=1200,
    step=12,
    help="Length of retirement in months.\n\nIn Guidance Mode, this should be the remaining number of months, if retirement is already underway."
)
analysis_start_date = st.sidebar.date_input(
    "Historical Analysis Start Date",
    value=datetime.date(1871, 1, 1),
    min_value=datetime.date(1871, 1, 1),
    max_value=datetime.date.today(),
    help="Earliest date for historical data included to estimate success rates (Shiller data begins in 1871).\n\nNote that when running historical "
         "simulations, each month's guardrails will be recalculated based on the historical data available between this start date and that month "
         "in history. A financial advisor running this strategy in the past would not have had a crystal ball to look into the future!"
)

# Numeric Inputs
initial_value = st.sidebar.number_input(
    "Initial Portfolio Value",
    value=1_000_000, min_value=100_000, step=100_000,
    help="Starting portfolio balance in dollars at retirement."
)

stock_pct = st.sidebar.slider(
    "Stock Percentage", value=0.75, min_value=0.0, max_value=1.0, step=0.05,
    help="Fraction of the portfolio allocated to US stocks; remainder to 10Y treasuries."
)

# Ensure widget defaults for existing cashflows remain synchronized prior to rendering inputs
def _ensure_cashflow_widget_state(idx: int, flow: dict) -> None:
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


def _sync_cashflows_from_widgets() -> None:
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


cap_options = ["Unlimited"] + [f"{pct}%" for pct in range(100, 201, 5)]
floor_options = ["Unlimited"] + [f"{pct}%" for pct in range(100, 24, -5)]

_sync_cashflows_from_widgets()

cashflows = _sanitize_cashflows(st.session_state.get("cashflows"))

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
    'cashflows': _cashflows_to_tuple(cashflows),
}
iwr_label_suffix = ""
try:
    if 'iwr_params' not in st.session_state or st.session_state['iwr_params'] != iwr_params:
        # Ensure Shiller data is loaded or cached in session_state
        shiller_df = st.session_state.get('shiller_df')
        if shiller_df is None:
            shiller_df = utils.load_shiller_data()
            st.session_state['shiller_df'] = shiller_df

        # Determine horizon length in months based on duration input
        if is_guidance:
            latest_shiller_date = pd.to_datetime(shiller_df["Date"].max())
            asof_for_iwr = latest_shiller_date if latest_shiller_date <= pd.to_datetime(datetime.date.today()) else pd.to_datetime(datetime.date.today())
            num_months = int(retirement_duration_months)
            if num_months <= 0:
                num_months = 360
            analysis_end_date_used = asof_for_iwr
        else:
            num_months = int(retirement_duration_months)
            if num_months <= 0:
                raise ValueError("Invalid retirement duration to compute initial WR.")
            analysis_end_date_used = iwr_params['start_date']

        res = utils.get_wr_for_fixed_success_rate(
            df=shiller_df,
            desired_success_rate=iwr_params['desired_success_rate'],
            num_months=num_months,
            analysis_start_date=iwr_params['analysis_start_date'],
            analysis_end_date=analysis_end_date_used,  # Use 'today' in Guidance Mode to match label logic
            initial_value=iwr_params['initial_value'],
            stock_pct=iwr_params['stock_pct'],
            tolerance=0.001,
            max_iterations=50,
            verbose=False,
            cashflows=cashflows,
        )
        st.session_state['iwr_value'] = float(res['withdrawal_rate']) if res['withdrawal_rate'] is not None else None
        st.session_state['iwr_params'] = iwr_params

    iwr = st.session_state.get('iwr_value')
    if iwr is not None:
        iwr_label_suffix = f" (Initial WR: {iwr*100:.2f}%)"
        auto_current_spending = round(float(initial_value) * float(iwr) / 12.0)
    else:
        iwr_label_suffix = " (Initial WR: N/A)"
        auto_current_spending = None
except Exception:
    iwr_label_suffix = " (Initial WR: N/A)"
    auto_current_spending = None
target_success_label = f"Target Success Rate{iwr_label_suffix}"
target_success_rate = st.sidebar.slider(
    target_success_label, value=st.session_state.get("target_success_rate", 0.90), min_value=0.0, max_value=1.0, step=0.01,
    help="Desired probability of success that will be used to select an initial withdrawal rate.\n\nThe initial "
         "withdrawal rate will be the rate at which fixed withdrawals over all periods of time with length = the "
         "configured retirement period length, between the Historical Analysis Start Date and the Retirement Start "
         "Date, end with >0 values this percent of the time.\n\nSetting this higher, e.g. 0.80-0.99, is more conservative: "
         "lower initial spending, lower chance of adjustment; setting this lower is more aggressive, 0.75-0.60 provides higher "
         "initial spending, higher chance of adjustment.",
    key="target_success_rate"
)

# Current spending input is primarily used in Guidance Mode but remains visible in Simulation Mode
if auto_current_spending is not None:
    st.session_state["_current_spending_auto_value"] = auto_current_spending
    current_value = st.session_state.get("current_monthly_spending")
    if st.session_state.get("_current_spending_overridden") and current_value is not None and np.isclose(
        float(current_value), float(auto_current_spending), rtol=0.0, atol=0.5
    ):
        st.session_state["_current_spending_overridden"] = False
    if not st.session_state.get("_current_spending_overridden"):
        if current_value is None or not np.isclose(float(current_value), float(auto_current_spending), rtol=0.0, atol=0.5):
            st.session_state["current_monthly_spending"] = float(auto_current_spending)
else:
    st.session_state["_current_spending_auto_value"] = None

if "current_monthly_spending" not in st.session_state or st.session_state["current_monthly_spending"] is None:
    st.session_state["current_monthly_spending"] = 0.0

current_monthly_spending = st.sidebar.number_input(
    "Current Monthly Spending",
    min_value=0.0,
    step=10.0,
    format="%.0f",
    help="Your current monthly spending level. Used only in Guidance Mode to compute guardrail values and hypothetical adjustments.",
    key="current_monthly_spending",
    on_change=_mark_current_spending_overridden,
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
        'cashflows': _cashflows_to_tuple(cashflows),
    }

    if ('guardrail_params' not in st.session_state) or (st.session_state['guardrail_params'] != gr_params):
        # Ensure Shiller data is available
        shiller_df = st.session_state.get('shiller_df')
        if shiller_df is None:
            shiller_df = utils.load_shiller_data()
            st.session_state['shiller_df'] = shiller_df

        # Determine horizon length in months based on duration input
        if is_guidance:
            latest_shiller_date = pd.to_datetime(shiller_df["Date"].max())
            asof_for_gr = latest_shiller_date if latest_shiller_date <= pd.to_datetime(datetime.date.today()) else pd.to_datetime(datetime.date.today())
            num_months = int(retirement_duration_months)
            if num_months <= 0:
                num_months = 360
            analysis_end_date_used = asof_for_gr
        else:
            num_months = int(retirement_duration_months)
            if num_months <= 0:
                raise ValueError("Invalid retirement duration to compute guardrail labels.")
            analysis_end_date_used = gr_params['start_date']
        if gr_params['iwr'] is None:
            raise ValueError("Initial withdrawal rate unavailable for guardrail label calculation.")

        # Initial withdrawal rate and first-period spending (already computed above for target label)
        first_month_spending = gr_params['initial_value'] * gr_params['iwr'] / 12.0

        # Compute WRs at start of retirement using retirement start date as analysis end date
        upper_res = utils.get_wr_for_fixed_success_rate(
            df=shiller_df,
            desired_success_rate=gr_params['upper_sr'],
            num_months=num_months,
            analysis_start_date=gr_params['analysis_start_date'],
            analysis_end_date=analysis_end_date_used,
            initial_value=gr_params['initial_value'],
            stock_pct=gr_params['stock_pct'],
            tolerance=0.001,
            max_iterations=50,
            verbose=False,
            cashflows=cashflows,
        )
        lower_res = utils.get_wr_for_fixed_success_rate(
            df=shiller_df,
            desired_success_rate=gr_params['lower_sr'],
            num_months=num_months,
            analysis_start_date=gr_params['analysis_start_date'],
            analysis_end_date=analysis_end_date_used,
            initial_value=gr_params['initial_value'],
            stock_pct=gr_params['stock_pct'],
            tolerance=0.001,
            max_iterations=50,
            verbose=False,
            cashflows=cashflows,
        )

        upper_wr = float(upper_res['withdrawal_rate']) if upper_res['withdrawal_rate'] is not None else None
        lower_wr = float(lower_res['withdrawal_rate']) if lower_res['withdrawal_rate'] is not None else None

        upper_pv = first_month_spending / upper_wr * 12 if (upper_wr is not None and upper_wr > 0) else None
        lower_pv = first_month_spending / lower_wr * 12 if (lower_wr is not None and lower_wr > 0) else None

        st.session_state['upper_label_suffix'] = f" (Initial PV: ${upper_pv:,.0f})" if upper_pv is not None else " (Initial PV: N/A)"
        st.session_state['lower_label_suffix'] = f" (Initial PV: ${lower_pv:,.0f})" if lower_pv is not None else " (Initial PV: N/A)"
        st.session_state['guardrail_params'] = gr_params

    upper_label_suffix = st.session_state.get('upper_label_suffix', " (Initial PV: N/A)")
    lower_label_suffix = st.session_state.get('lower_label_suffix', " (Initial PV: N/A)")
except Exception:
    upper_label_suffix = " (Initial PV: N/A)"
    lower_label_suffix = " (Initial PV: N/A)"

upper_guardrail_label = f"Upper Guardrail Success Rate{upper_label_suffix}"
lower_guardrail_label = f"Lower Guardrail Success Rate{lower_label_suffix}"

upper_guardrail_success = st.sidebar.slider(
    upper_guardrail_label, value=st.session_state.get("upper_guardrail_success", 1.00), min_value=0.0, max_value=1.0, step=0.01,
    help="The withdrawal rate used to calculate the upper guardrail portfolio value.\n\nThis is the value where the "
         "current withdrawal amount, if held constant, will succeed this frequently or more, for all periods with "
         "length = # months remaining in retirement, between the Historical Analysis Start Date and the current "
         "simulation date.\n\nSetting this higher is more conservative, and will cause you to wait longer to increase "
         "your spending when markets are up.",
    key="upper_guardrail_success"
)
lower_guardrail_success = st.sidebar.slider(
    lower_guardrail_label, value=st.session_state.get("lower_guardrail_success", 0.75), min_value=0.0, max_value=1.0, step=0.01,
    help="The withdrawal rate used to calculate the lower guardrail portfolio value.\n\nThis is the value where the "
         "current withdrawal amount, if held constant, will succeed this frequently or less, for all periods with "
         "length = # months remaining in retirement, between the Historical Analysis Start Date and the current "
         "simulation date.\n\nSetting this higher is more conservative, and will cause you to decrease your spending "
         "sooner when markets are down.",
    key="lower_guardrail_success"
)
upper_adjustment_fraction = st.sidebar.slider(
    "Upper Adjustment Fraction", value=1.0, min_value=0.0, max_value=1.0, step=0.05,
    help="How much to increase spending when we hit the upper guardrail.\n\nExpressed as a % of the distance between "
         "the Upper Guardrail Success Rate and the Target Success Rate.\n\nFor example, if the upper guardrail "
         "represents 100% success and the target is 90%, setting this value to 50% means we go half the distance back "
         "to the target, and our new withdrawal rate will be based on a 95% chance of success.\n\nSetting this higher "
         "is more aggressive, and will cause you to make larger spending increases when you hit the upper guardrail."
)
lower_adjustment_fraction = st.sidebar.slider(
    "Lower Adjustment Fraction", value=0.1, min_value=0.0, max_value=1.0, step=0.05,
    help="How much to decrease spending when we hit the lower guardrail.\n\nExpressed as a % of the distance between "
         "the Lower Guardrail Success Rate and the Target Success Rate.\n\nFor example, if the lower guardrail "
         "represents 70% success and the target is 90%, setting this value to 50% means we go half the distance back "
         "to the target, and our new withdrawal rate will be based on an 80% chance of success.\n\nSetting this higher "
         "is more conservative, and will cause you to make larger spending decreases when you hit the lower guardrail."
)

adjustment_threshold = st.sidebar.slider(
    "Adjustment Threshold (e.g., 0.05 for 5%)",
    value=0.0 if is_guidance else 0.05,
    min_value=0.0, max_value=0.2, step=0.01,
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
         "and any resulting spending changes to the beginning of those periods (Jan/Apr/Jul/Oct, Jan/Jul, or January)."
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
    'cashflows': _cashflows_to_tuple(cashflows),
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


def render_simulation_results(results_df: pd.DataFrame) -> None:
    """Render charts and summaries for simulation results."""

    if results_df is None or results_df.empty:
        st.info("No simulation results to display.")
        return

    show_guardrail_hits = st.checkbox(
        "Show guardrail hit markers",
        value=True,
        help="Toggle vertical dotted lines at guardrail hits.",
        key="show_guardrail_hits"
    )

    init_withdrawal = float(results_df['Withdrawal'].iloc[0]) if not results_df.empty else 0.0

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.14,
        row_heights=[0.55, 0.45],
        subplot_titles=("Portfolio Value vs Guardrails", "Withdrawals Over Time")
    )

    if 'Fixed_WR_Value' in results_df.columns:
        fig.add_trace(
            go.Scatter(
                x=results_df['Date'],
                y=results_df['Fixed_WR_Value'],
                mode='lines',
                name='Value w/Fixed WR',
                line=dict(color='#7f7f7f'),
                opacity=0.6,
                hovertemplate='<b>%{fullData.name}</b>: $%{y:,.2f}<extra></extra>'
            ),
            row=1,
            col=1
        )

    fig.add_trace(
        go.Scatter(
            x=results_df['Date'],
            y=results_df['Upper_Guardrail'],
            mode='lines',
            name='Upper Guardrail',
            line=dict(color='#2ca02c'),
            opacity=0.45,
            hovertemplate='<b>%{fullData.name}</b>: $%{y:,.2f}<extra></extra>'
        ),
        row=1,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=results_df['Date'],
            y=results_df['Lower_Guardrail'],
            mode='lines',
            name='Lower Guardrail',
            line=dict(color='#d62728'),
            opacity=0.45,
            hovertemplate='<b>%{fullData.name}</b>: $%{y:,.2f}<extra></extra>'
        ),
        row=1,
        col=1
    )
    fig.add_trace(
        go.Scatter(
            x=results_df['Date'],
            y=results_df['Portfolio_Value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#1f77b4'),
            hovertemplate='<b>%{fullData.name}</b>: $%{y:,.2f}<extra></extra>'
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=results_df['Date'],
            y=results_df['Withdrawal'],
            mode='lines',
            name='Withdrawal',
            line=dict(color='#9467bd'),
            hovertemplate='<b>%{fullData.name}</b>: $%{y:,.2f}<extra></extra>'
        ),
        row=2,
        col=1
    )
    if 'Net_Cashflow' in results_df.columns:
        fig.add_trace(
            go.Scatter(
                x=results_df['Date'],
                y=results_df['Net_Cashflow'],
                mode='lines',
                name='Net Cashflow',
                line=dict(color='#17becf', dash='dot'),
                hovertemplate='<b>%{fullData.name}</b>: $%{y:,.2f}<extra></extra>'
            ),
            row=2,
            col=1
        )
    if 'Total_Income' in results_df.columns:
        fig.add_trace(
            go.Scatter(
                x=results_df['Date'],
                y=results_df['Total_Income'],
                mode='lines',
                name='Total Income',
                line=dict(color='#bcbd22'),
                hovertemplate='<b>%{fullData.name}</b>: $%{y:,.2f}<extra></extra>'
            ),
            row=2,
            col=1
        )
    fig.add_trace(
        go.Scatter(
            x=results_df['Date'],
            y=results_df['Fixed_WR_Withdrawal'] if 'Fixed_WR_Withdrawal' in results_df.columns else [init_withdrawal] * len(results_df),
            mode='lines',
            name='Initial Withdrawal',
            line=dict(color='#7f7f7f', dash='dash'),
            hovertemplate='<b>%{fullData.name}</b>: $%{y:,.2f}<extra></extra>'
        ),
        row=2,
        col=1
    )

    shapes = []
    if show_guardrail_hits:
        def _guardrail_marker(date_value, color):
            return [
                dict(
                    type='line', xref='x', yref='y domain',
                    x0=date_value, x1=date_value, y0=0, y1=1,
                    line=dict(color=color, width=1, dash='dot'),
                    layer='below'
                ),
                dict(
                    type='line', xref='x', yref='y2 domain',
                    x0=date_value, x1=date_value, y0=0, y1=1,
                    line=dict(color=color, width=1, dash='dot'),
                    layer='below'
                )
            ]

        for d in results_df.loc[results_df['Guardrail_Hit'] == 'UPPER', 'Date']:
            shapes.extend(_guardrail_marker(d, '#2ca02c'))
        for d in results_df.loc[results_df['Guardrail_Hit'] == 'LOWER', 'Date']:
            shapes.extend(_guardrail_marker(d, '#d62728'))

    fig.update_layout(
        shapes=shapes,
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.16,
            xanchor='center',
            x=0.5,
            traceorder='reversed'
        ),
        showlegend=True,
        margin=dict(l=10, r=10, t=80, b=60),
        dragmode='zoom',
        height=900,
        xaxis=dict(rangeslider=dict(visible=False))
    )
    fig.update_annotations(
        font=dict(size=24),
        yshift=12,
        yanchor='bottom'
    )
    fig.update_xaxes(
        type='date',
        hoverformat='%b %d, %Y',
        showticklabels=True,
        row=1,
        col=1
    )
    fig.update_xaxes(
        title_text='Date',
        type='date',
        hoverformat='%b %d, %Y',
        rangeslider=dict(visible=False),
        row=2,
        col=1
    )
    fig.update_yaxes(
        title_text='Value ($)',
        tickprefix='$',
        tickformat=',.0f',
        automargin=True,
        rangemode='tozero',
        row=1,
        col=1
    )
    fig.update_yaxes(
        title_text='Withdrawals & Cashflows ($/month)',
        tickprefix='$',
        tickformat=',.0f',
        automargin=True,
        rangemode='tozero',
        row=2,
        col=1
    )

    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': False})

    total_fixed = float(results_df['Fixed_WR_Withdrawal'].sum()) if 'Fixed_WR_Withdrawal' in results_df.columns else float(init_withdrawal) * len(results_df)
    total_guardrails = float(results_df['Withdrawal'].sum())
    total_cashflow = float(results_df['Net_Cashflow'].sum()) if 'Net_Cashflow' in results_df.columns else 0.0
    total_income_guardrails = float(results_df['Total_Income'].sum()) if 'Total_Income' in results_df.columns else total_guardrails + total_cashflow
    total_income_fixed = float(results_df['Fixed_WR_Total_Income'].sum()) if 'Fixed_WR_Total_Income' in results_df.columns else total_fixed + total_cashflow
    withdrawal_diff_ratio = (total_guardrails - total_fixed) / total_fixed if total_fixed else None
    income_diff_ratio = (total_income_guardrails - total_income_fixed) / total_income_fixed if total_income_fixed else None

    if 'Total_Income' in results_df.columns:
        income_series = results_df['Total_Income'].astype(float)
    else:
        income_series = results_df['Withdrawal'].astype(float) if 'Withdrawal' in results_df.columns else pd.Series(dtype=float)
        if 'Net_Cashflow' in results_df.columns and not income_series.empty:
            income_series = income_series + results_df['Net_Cashflow'].astype(float)

    start_income = float(income_series.iloc[0]) if not income_series.empty else None
    min_income = float(income_series.min()) if not income_series.empty else None
    max_income = float(income_series.max()) if not income_series.empty else None

    def _longest_run(values, target):
        longest = 0
        current = 0
        for val in values:
            if np.isclose(val, target, rtol=1e-9, atol=0.5):
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        return longest

    min_streak = _longest_run(income_series, min_income) if not income_series.empty else 0
    max_streak = _longest_run(income_series, max_income) if not income_series.empty else 0

    def _fmt_currency(value):
        if value is None or not np.isfinite(value):
            return "N/A"
        return f"${value:,.0f}"

    def _fmt_pct_diff(new_value, baseline):
        if new_value is None or baseline in (None, 0):
            return "N/A"
        return f"{(new_value / baseline - 1.0):+.0%}"

    summary_rows = [
        {
            "Metric": "Total Withdrawal",
            "Fixed": _fmt_currency(total_fixed),
            "Guardrails": _fmt_currency(total_guardrails),
            "% Diff": _fmt_pct_diff(total_guardrails, total_fixed) if withdrawal_diff_ratio is not None else "N/A",
        },
        {
            "Metric": "Total Income",
            "Fixed": _fmt_currency(total_income_fixed),
            "Guardrails": _fmt_currency(total_income_guardrails),
            "% Diff": _fmt_pct_diff(total_income_guardrails, total_income_fixed) if income_diff_ratio is not None else "N/A",
        },
    ]

    st.table(pd.DataFrame(summary_rows).set_index("Metric"))

    income_rows = [
        {
            "Metric": "Start Income",
            "Monthly": _fmt_currency(start_income),
            "% Diff": "—",
            "Duration (months)": "—",
        },
        {
            "Metric": "Min Income",
            "Monthly": _fmt_currency(min_income),
            "% Diff": _fmt_pct_diff(min_income, start_income),
            "Duration (months)": f"{min_streak}" if min_streak else "—",
        },
        {
            "Metric": "Max Income",
            "Monthly": _fmt_currency(max_income),
            "% Diff": _fmt_pct_diff(max_income, start_income),
            "Duration (months)": f"{max_streak}" if max_streak else "—",
        },
    ]

    st.table(pd.DataFrame(income_rows).set_index("Metric"))


with st.sidebar.expander("Advanced Controls"):
    st.selectbox(
        "Spending Cap",
        options=cap_options,
        key="spending_cap_option",
        help="Maximum spending level as a percent of the initial monthly spending.",
    )
    st.selectbox(
        "Spending Floor",
        options=floor_options,
        key="spending_floor_option",
        help="Minimum spending level as a percent of the initial monthly spending.",
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
        _ensure_cashflow_widget_state(idx, flow)

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
            _clear_cashflow_widget_state(idx)
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
            format="%0.2f",
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
            current_monthly_spending=st.session_state.get("current_monthly_spending", 40000.0),
            stock_pct=stock_pct,
            target_success_rate=target_success_rate,
            upper_guardrail_success=upper_guardrail_success,
            lower_guardrail_success=lower_guardrail_success,
            upper_adjustment_fraction=upper_adjustment_fraction,
            lower_adjustment_fraction=lower_adjustment_fraction,
            adjustment_frequency=adjustment_frequency,
            spending_cap=spending_cap_multiplier,
            spending_floor=spending_floor_multiplier,
            cashflows=cashflows,
            verbose=False
        )

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

        adjustments_allowed = snap.get("adjustments_allowed", True)
        next_adjustment_date = snap.get("next_adjustment_date")
        cashflow_month0 = snap.get("current_cashflow")

        def fmt_month(ts):
            if ts is None or pd.isna(ts):
                return "N/A"
            timestamp = pd.to_datetime(ts)
            return timestamp.strftime("%B %Y")

        st.subheader("Guidance Mode")
        st.markdown("Use this mode to generate forward-looking guidance for a client who is retired today and in drawdown.\n\n"
                    "For more information, see the [official documentation](https://github.com/rogercost/fire-guardrails/blob/main/README.md).")

        if not adjustments_allowed:
            st.info(
                "Adjustments are restricted to the selected cadence. "
                f"The next eligible adjustment month is {fmt_month(next_adjustment_date)}."
            )

        if start_wr is not None:
            st.markdown(
                f"* **Target Withdrawal Rate:** {start_wr*100:.2f}% "
                f"({fmt_money(start_month)}/month or {fmt_money(start_year)}/year total spending based on the Initial Portfolio Value) "
                f"— portfolio withdrawal after cashflows: {fmt_money(start_net_withdrawal)}/month"
            )
        else:
            st.markdown("**Starting Withdrawal Rate:** N/A")

        st.markdown(f"* **Month 1 Cashflows:** {fmt_money(cashflow_month0)}/month")
        if adjustments_allowed:
            st.markdown(
                f"* **Upper Guardrail Portfolio Value:** {fmt_money(upper_val)} based on the Current Monthly Spending\n  * If client's portfolio value exceeds this, adjust "
                f"spending by {fmt_pct(up_adj_pct)} to {fmt_money(up_adj_month)}/month or {fmt_money(up_adj_year)}/year"
            )
            st.markdown(
                f"* **Lower Guardrail Portfolio Value:** {fmt_money(lower_val)} based on the Current Monthly Spending\n  * If client's portfolio value falls below this, adjust "
                f"spending by {fmt_pct(low_adj_pct)} to {fmt_money(low_adj_month)}/month or {fmt_money(low_adj_year)}/year"
            )
        else:
            st.markdown(
                f"* **Upper Guardrail Portfolio Value:** {fmt_money(upper_val)} based on the Current Monthly Spending."
            )
            st.markdown(
                f"* **Lower Guardrail Portfolio Value:** {fmt_money(lower_val)} based on the Current Monthly Spending."
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

    render_simulation_results(results_df)

elif not is_guidance:
    if 'results_df' in st.session_state:
        results_df = st.session_state['results_df']

        if st.session_state.get('dirty'):
            render_dirty_banner()

        render_simulation_results(results_df)
    else:
        st.subheader("Simulation Mode")
        st.markdown("Use this mode to simulate running a guardrail-based retirement withdrawal strategy during a historical period.\n\n"
                    "For more information, see the [official documentation](https://github.com/rogercost/fire-guardrails/blob/main/README.md).")
        st.info("Adjust parameters in the sidebar and click 'Run Simulation' to start.")
