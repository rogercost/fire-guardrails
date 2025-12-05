import datetime
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

import controls
import display
import utils
from app_settings import CashflowSetting, Settings

st.set_page_config(layout="wide", page_title="Guardrail Withdrawal Simulator")

# Apply configuration from hyperlink when available (only once per session)
if "_settings_initialized" not in st.session_state:
    controls.hydrate_settings()

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

controls.initialize_display()

def _render_sidebar_label(text: str, color: Optional[str] = None) -> None:
    """Render styled sidebar helper text with optional highlighting."""

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

settings_error = st.session_state.get("_settings_error")
if settings_error:
    st.sidebar.error(f"Unable to load shared configuration: {settings_error}")

# Date Inputs
today_date = datetime.date.today()
start_date_default = datetime.date(1968, 4, 1)
start_date_key = "retirement_start_date"
sim_start_key = "_simulation_retirement_start_date"
controls.init_start_date_field(today_date, sim_start_key, start_date_key, start_date_default, mode, is_guidance)

start_date = st.sidebar.date_input(
    "Retirement Start Date",
    min_value=datetime.date(1871, 1, 1),
    max_value=today_date,
    help="First retirement date used for withdrawals.\n\nCurrently, only the year and month are used, due to the monthly nature of the Shiller "
         "dataset. The day of the month is ignored.\n\nIn Simulation Mode, the historical simulations begin from this date.\n\n"
         "In Guidance Mode, this defaults to today. Even if retirement is already underway, the guidance is forward looking from today.\n\n"
         "(Hint: You can type the date in YYYY/MM/DD format instead of choosing it from the selector, which may be faster.)",
    disabled=is_guidance,  # In Guidance Mode, this is fixed to today
    on_change=_unmark_initial_spending_overridden,
    key=start_date_key,
)

if is_guidance:
    start_date = today_date
else:
    start_date = controls.get_date_state(start_date_key, start_date_default)
    st.session_state[sim_start_key] = start_date

retirement_duration_months = st.sidebar.number_input(
    "Retirement Duration (months)",
    value=controls.get_int_state("retirement_duration_months", 360),
    min_value=1,
    max_value=1200,
    step=12,
    on_change=_unmark_initial_spending_overridden,
    key="retirement_duration_months",
    help="Length of retirement in months.\n\nIn Guidance Mode, this should be the remaining number of months, if retirement is already underway."
)

analysis_start_date = st.sidebar.date_input(
    "Historical Analysis Start Date",
    value=controls.get_date_state("analysis_start_date", datetime.date(1871, 1, 1)),
    min_value=datetime.date(1871, 1, 1),
    max_value=datetime.date.today(),
    on_change=_unmark_initial_spending_overridden,
    help="Earliest date for historical data included to estimate success rates (Shiller data begins in 1871).\n\nCurrently, only the year and month "
         "are used, due to the monthly nature of the Shiller dataset. The day of the month is ignored.\n\nNote that when running historical "
         "simulations, each month's guardrails will be recalculated based on the historical data available between this start date and that month "
         "in history. A financial advisor running this strategy in the past would not have had a crystal ball to look into the future!\n\n"
         "(Hint: You can type the date in YYYY/MM/DD format instead of choosing it from the selector, which may be faster.)",
    key="analysis_start_date",
)

# Numeric Inputs
initial_value = st.sidebar.number_input(
    "Initial Portfolio Value",
    value=controls.get_float_state("initial_portfolio_value", 1_000_000.0),
    min_value=100_000.0,
    step=100_000.0,
    on_change=_unmark_initial_spending_overridden,
    key="initial_portfolio_value",
    help="Starting portfolio balance in dollars at retirement."
)

stock_pct = st.sidebar.slider(
    "Stock Percentage",
    value=controls.get_float_state("stock_pct", 0.75),
    min_value=0.0,
    max_value=1.0,
    step=0.01,
    on_change=_unmark_initial_spending_overridden,
    key="stock_pct",
    help="Fraction of the portfolio allocated to US stocks; remainder to 10Y treasuries."
)

cap_options = ["Unlimited"] + [f"{pct}%" for pct in range(100, 201, 5)]
floor_options = ["Unlimited"] + [f"{pct}%" for pct in range(100, 24, -5)]

controls.sync_cashflows_from_widgets()

sanitized_cashflows = controls.sanitize_cashflows(st.session_state.get("cashflows"))

# Compute Initial Spending Rate (ISR) to show in the Target Success Rate label.
# We recompute only when relevant inputs change to avoid unnecessary calls.
isr_params = {
    'start_date': pd.to_datetime(start_date),
    'duration_months': int(retirement_duration_months),
    'analysis_start_date': pd.to_datetime(analysis_start_date),
    'initial_value': float(initial_value),
    'stock_pct': float(stock_pct),
    # Use current target_success_rate if available, otherwise default of 0.90
    'desired_success_rate': float(st.session_state.get('target_success_rate', 0.90)),
    'final_value_target': float(st.session_state.get('final_value_target', 0.0)),
    'cashflows': controls.cashflows_to_tuple(sanitized_cashflows),
}
isr_label_suffix = ""

try:
    if 'isr_params' not in st.session_state or st.session_state['isr_params'] != isr_params:
        display.update_isr_dynamic_label(isr_params=isr_params, is_guidance=is_guidance, cashflows=sanitized_cashflows)
    isr = st.session_state.get('isr_value')
    if isr is not None:
        isr_label_suffix = f" (Initial SR: {isr*100:.2f}%)"
        auto_initial_spending = round(float(initial_value) * float(isr) / 12.0)
    else:
        isr_label_suffix = " (Initial SR: N/A)"
        auto_initial_spending = None

except Exception as e:
    print(e)
    isr_label_suffix = " (Initial SR: N/A)"
    auto_initial_spending = None

target_success_label = f"Target Success Rate{isr_label_suffix}"
target_success_rate = st.sidebar.slider(
    target_success_label,
    value=st.session_state.get("target_success_rate", 0.90),
    min_value=0.0,
    max_value=1.0,
    step=0.01,
    help="Desired probability of success that will be used to select an initial spending rate.\n\nThe initial "
         "spending rate will be the rate at which fixed spending over all periods of time with length = the "
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
         "Automatically updates as you change the Target Spending Rate, but can be set to a custom value if desired.\n\n",
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
        'isr': float(st.session_state.get('isr_value')) if st.session_state.get('isr_value') is not None else None,
        'initial_spending': float(st.session_state.get("initial_monthly_spending", 0.0)),
        'final_value_target': float(st.session_state.get('final_value_target', 0.0)),
        'cashflows': controls.cashflows_to_tuple(sanitized_cashflows),
    }

    if ('guardrail_params' not in st.session_state) or (st.session_state['guardrail_params'] != gr_params):
        display.update_guardrail_dynamic_labels(gr_params=gr_params, is_guidance=is_guidance, cashflows=sanitized_cashflows)

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
    help="The spending rate used to calculate the upper guardrail portfolio value.\n\nThis is the value where the "
         "current spending amount, if held constant, will succeed this frequently or more, for all periods with "
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
    help="The spending rate used to calculate the lower guardrail portfolio value.\n\nThis is the value where the "
         "current spending amount, if held constant, will succeed this frequently or less, for all periods with "
         "length = # months remaining in retirement, between the Historical Analysis Start Date and the current "
         "simulation date.\n\nSetting this higher is more conservative, and will cause you to decrease your spending "
         "sooner when markets are down.",
    key="lower_guardrail_success",
    label_visibility="collapsed"
)

upper_adjustment_fraction = st.sidebar.slider(
    "Upper Adjustment Fraction",
    value=controls.get_float_state("upper_adjustment_fraction", 1.0),
    min_value=0.0,
    max_value=1.0,
    step=0.05,
    key="upper_adjustment_fraction",
    help="How much to increase spending when we hit the upper guardrail.\n\nExpressed as a % of the distance between "
         "the Upper Guardrail Success Rate and the Target Success Rate.\n\nFor example, if the upper guardrail "
         "represents 100% success and the target is 90%, setting this value to 50% means we go half the distance back "
         "to the target, and our new spending rate will be based on a 95% chance of success.\n\nSetting this higher "
         "is more aggressive, and will cause you to make larger spending increases when you hit the upper guardrail."
)

lower_adjustment_fraction = st.sidebar.slider(
    "Lower Adjustment Fraction",
    value=controls.get_float_state("lower_adjustment_fraction", 0.1),
    min_value=0.0,
    max_value=1.0,
    step=0.05,
    key="lower_adjustment_fraction",
    help="How much to decrease spending when we hit the lower guardrail.\n\nExpressed as a % of the distance between "
         "the Lower Guardrail Success Rate and the Target Success Rate.\n\nFor example, if the lower guardrail "
         "represents 70% success and the target is 90%, setting this value to 50% means we go half the distance back "
         "to the target, and our new spending rate will be based on an 80% chance of success.\n\nSetting this higher "
         "is more conservative, and will cause you to make larger spending decreases when you hit the lower guardrail."
)

default_threshold = 0.0 if is_guidance else 0.05
adjustment_threshold = st.sidebar.slider(
    "Adjustment Threshold (e.g., 0.05 for 5%)",
    value=controls.get_float_state("adjustment_threshold", default_threshold),
    min_value=0.0,
    max_value=0.2,
    step=0.01,
    key="adjustment_threshold",
    help="The minimum percent difference between our new spending and our prior spending, before we make a change.\n\n"
         "Even if we hit a guardrail, we may elect to set this to 5% to avoid making lots of small adjustments. Set it "
         "to 0% to disable it and allow all guardrail hits to trigger spending adjustments.\n\nSetting this higher is "
         "neither aggressive nor conservative, since it impacts spending increases as well as decreases. It is purely "
         "a question of whether frequent adjustments are acceptable and administratively feasible for the client.\n\n"
         "Not used in Guidance Mode, which will always show spending adjustment suggestions, no matter how small.",
    disabled=is_guidance  # In Guidance Mode, single-run snapshot ignores threshold
)

frequency_options = ["Monthly", "Quarterly", "Biannually", "Annually"]
current_frequency = st.session_state.get("adjustment_frequency", "Monthly")
try:
    frequency_index = frequency_options.index(str(current_frequency))
except ValueError:
    frequency_index = 0
adjustment_frequency = st.sidebar.selectbox(
    "Adjustment Frequency",
    options=frequency_options,
    index=frequency_index,
    key="adjustment_frequency",
    help="How often spending adjustments are permitted. Choosing Quarterly, Biannually, or Annually restricts guardrail checks "
         "and any resulting spending changes to the beginning of those periods (Jan/Apr/Jul/Oct, Jan/Jul, or January).",
    disabled=is_guidance  # In Guidance Mode, no decision gating, it's up to the adviser and client
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
    st.number_input(
        "Final Value Target (Bequest)",
        value=controls.get_float_state("final_value_target", 0.0),
        min_value=0.0,
        max_value=float(initial_value),
        step=10000.0,
        format="%.0f",
        key="final_value_target",
        help="Minimum ending portfolio value required for a simulation to be considered successful. "
             "Set to $0 for no bequest requirement. When set higher, the simulation will count "
             "as a failure if the portfolio ends below this amount, even if it never depleted. "
             "This affects both the success rate calculation and the recommended safe spending rate.\n\n"
             "Note that setting this too high may result in huge upper guardrail values - particularly "
             "if the portfolio value during retirement crosses below this value, as the required spending "
             "% rate (the denominator of the calculation) for the portfolio to end higher than its current "
             "value approaches zero.",
    )

    if st.button("Add Recurring Cashflow", key="add_cashflow_btn"):
        st.session_state["cashflows"].append({
            "start_month": 0,
            "end_month": 0,
            "amount": 0.0,
            "label": f"Cashflow {len(st.session_state['cashflows']) + 1}",
        })
        st.rerun()

    controls.draw_cashflow_widget_rows()


# Build Settings object representing the full control state
cashflow_settings = [
    cf for cf in (CashflowSetting.from_dict(flow) for flow in sanitized_cashflows)
    if cf is not None
]

settings = Settings(
    mode=mode,
    start_date=start_date,
    retirement_duration_months=int(retirement_duration_months),
    analysis_start_date=analysis_start_date,
    initial_value=float(initial_value),
    stock_pct=float(stock_pct),
    target_success_rate=float(target_success_rate),
    initial_monthly_spending=float(initial_monthly_spending),
    initial_spending_overridden=bool(st.session_state.get("_initial_spending_overridden", False)),
    upper_guardrail_success=float(upper_guardrail_success),
    lower_guardrail_success=float(lower_guardrail_success),
    upper_adjustment_fraction=float(upper_adjustment_fraction),
    lower_adjustment_fraction=float(lower_adjustment_fraction),
    adjustment_threshold=float(adjustment_threshold),
    adjustment_frequency=adjustment_frequency,
    spending_cap_option=st.session_state.get("spending_cap_option", "Unlimited"),
    spending_floor_option=st.session_state.get("spending_floor_option", "Unlimited"),
    final_value_target=float(st.session_state.get("final_value_target", 0.0)),
    cashflows=cashflow_settings,
)

st.session_state["settings"] = settings

# Warn if guardrail success rates are in unexpected order
if not (lower_guardrail_success <= target_success_rate <= upper_guardrail_success):
    st.sidebar.warning(
        "Guardrail success rates are in an unusual order. Typically: "
        "Lower Guardrail \u2264 Target \u2264 Upper Guardrail. "
        "Current values may produce unexpected behavior."
    )

encoded_config = settings.to_base64()
st.session_state["_encoded_settings"] = encoded_config
share_link_url = f"?config={encoded_config}"

sim_signature = settings.simulation_signature()
last_run_signature = st.session_state.get('last_run_signature')
dirty = last_run_signature is not None and last_run_signature != sim_signature
st.session_state['dirty'] = dirty

# When inputs change, visually dim and surround the main area with a red border
if dirty and not is_guidance:
    controls.draw_dirty_border()

# ------ Main Program Logic -------
#
if is_guidance:
    # Guidance Mode: run a single-iteration snapshot and display text output

    shiller_df = utils.get_cached_shiller_df(st.session_state)

    # Use the latest available Shiller data date (<= today) as the as-of date.
    today = datetime.date.today()
    latest_shiller_date = pd.to_datetime(shiller_df["Date"].max()).date()
    asof_date = latest_shiller_date if latest_shiller_date <= today else today

    try:
        snap = utils.compute_guardrail_guidance_snapshot(
            df=shiller_df,
            asof_date=asof_date,
            settings=settings,
        )

        display.render_guidance_results(snap=snap)

    except Exception as e:
        st.error(f"Unable to compute guidance snapshot: {e}")

elif st.sidebar.button(
    "Run Simulation",
    help="Fetch data and run the guardrail withdrawal simulation with the selected parameters."
):
    status_ph = st.empty()
    status_ph.text("Loading Shiller data...")
    shiller_df = utils.get_cached_shiller_df(st.session_state)
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
        settings=settings,
        verbose=True,
        on_progress=on_progress,
        on_status=on_status
    )

    # Cache results in session state for re-render without recomputation
    st.session_state['results_df'] = results_df
    st.session_state['last_run_signature'] = sim_signature
    st.session_state['dirty'] = False

    # Remove status line entirely to place plots at the very top
    status_ph.empty()

    display.render_simulation_results(results_df)

elif not is_guidance:
    if 'results_df' in st.session_state:
        results_df = st.session_state['results_df']

        if st.session_state.get('dirty'):
            controls.render_dirty_banner()

        display.render_simulation_results(results_df)
    else:
        st.subheader("Simulation Mode")
        st.markdown("Use this mode to simulate running a guardrail-based retirement withdrawal strategy during a historical period.\n\n"
                    "All dollar amounts shown are in real (constant) dollars, net of inflation. For more details, see the [documentation](https://github.com/rogercost/fire-guardrails/blob/main/README.md).")
        st.info("Adjust parameters in the sidebar and click 'Run Simulation' to start.")

st.divider()
st.markdown(f"[Shareable link to this run]({share_link_url})")
st.caption("Copy the link to load these settings on any device.")
