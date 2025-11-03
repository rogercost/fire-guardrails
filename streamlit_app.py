import datetime

import streamlit as st
import plotly.graph_objects as go
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

# New input used by Guidance Mode (hidden/disabled in Simulation Mode)
current_monthly_spending = st.sidebar.number_input(
    "Current Monthly Spending",
    value=3300,
    min_value=0,
    step=10,
    help="Your current monthly spending level. Used only in Guidance Mode to compute guardrail values and hypothetical adjustments.",
    disabled=not is_guidance,
    key="current_monthly_spending"
)

stock_pct = st.sidebar.slider(
    "Stock Percentage", value=0.75, min_value=0.0, max_value=1.0, step=0.05,
    help="Fraction of the portfolio allocated to US stocks; remainder to 10Y treasuries."
)

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
            verbose=False
        )
        st.session_state['iwr_value'] = float(res['withdrawal_rate']) if res['withdrawal_rate'] is not None else None
        st.session_state['iwr_params'] = iwr_params

    iwr = st.session_state.get('iwr_value')
    if iwr is not None:
        iwr_label_suffix = f" (Initial WR: {iwr*100:.2f}%)"
    else:
        iwr_label_suffix = " (Initial WR: N/A)"
except Exception:
    iwr_label_suffix = " (Initial WR: N/A)"
target_success_label = f"Target Success Rate{iwr_label_suffix}"
target_success_rate = st.sidebar.slider(
    target_success_label, value=st.session_state.get("target_success_rate", 0.90), min_value=0.0, max_value=1.0, step=0.01,
    help="Desired probability of success that will be used to select an initial withdrawal rate.\n\nThe initial "
         "withdrawal rate will be the rate at which fixed withdrawals over all periods of time with length = the "
         "configured retirement period length, between the Historical Analysis Start Date and the Retirement Start "
         "Date, end with >0 values this percent of the time.\n\nSetting this higher is more aggressive, and increases "
         "the probability that your first adjustment will be a decrease as opposed to an increase in spending.",
    key="target_success_rate"
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
            verbose=False
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
            verbose=False
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

        st.markdown(
            f"* **Target Withdrawal Rate:** {start_wr*100:.2f}% ({fmt_money(start_month)}/month or {fmt_money(start_year)}/year "
            f"based on the Initial Portfolio Value)"
            if start_wr is not None else
            "**Starting Withdrawal Rate:** N/A"
        )
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

    st.subheader("Simulation Mode")

    # Charts (table removed)
    st.subheader("Portfolio Value vs Guardrails")

    show_guardrail_hits = st.checkbox(
        "Show guardrail hit markers",
        value=True,
        help="Toggle vertical dotted lines at guardrail hits.",
        key="show_guardrail_hits"
    )

    fig1 = go.Figure()
    # Add fixed-withdrawal portfolio value first so it renders behind others
    if 'Fixed_WR_Value' in results_df.columns:
        fig1.add_trace(go.Scatter(
            x=results_df['Date'], y=results_df['Fixed_WR_Value'],
            mode='lines', name='Value w/Fixed WR', line=dict(color='#7f7f7f'),
            opacity=0.6,
            hovertemplate='<b>%{fullData.name}</b>: $%{y:,.2f}<extra></extra>'
        ))
    # Draw guardrails first (muted) so portfolio is visually on top
    fig1.add_trace(go.Scatter(
        x=results_df['Date'], y=results_df['Upper_Guardrail'],
        mode='lines', name='Upper Guardrail', line=dict(color='#2ca02c'), opacity=0.45,
        hovertemplate='<b>%{fullData.name}</b>: $%{y:,.2f}<extra></extra>'
    ))
    fig1.add_trace(go.Scatter(
        x=results_df['Date'], y=results_df['Lower_Guardrail'],
        mode='lines', name='Lower Guardrail', line=dict(color='#d62728'), opacity=0.45,
        hovertemplate='<b>%{fullData.name}</b>: $%{y:,.2f}<extra></extra>'
    ))
    fig1.add_trace(go.Scatter(
        x=results_df['Date'], y=results_df['Portfolio_Value'],
        mode='lines', name='Portfolio Value', line=dict(color='#1f77b4'),
        hovertemplate='<b>%{fullData.name}</b>: $%{y:,.2f}<extra></extra>'
    ))
    # Add thin vertical dotted lines at guardrail hits (drawn behind data)
    shapes = []
    if show_guardrail_hits:
        for d in results_df.loc[results_df['Guardrail_Hit'] == 'UPPER', 'Date']:
            shapes.append(dict(
                type='line', xref='x', yref='paper',
                x0=d, x1=d, y0=0, y1=1,
                line=dict(color='#2ca02c', width=1, dash='dot'),
                layer='below'
            ))
        for d in results_df.loc[results_df['Guardrail_Hit'] == 'LOWER', 'Date']:
            shapes.append(dict(
                type='line', xref='x', yref='paper',
                x0=d, x1=d, y0=0, y1=1,
                line=dict(color='#d62728', width=1, dash='dot'),
                layer='below'
            ))

    fig1.update_layout(
        shapes=shapes,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0, traceorder='reversed'),
        margin=dict(l=10, r=10, t=10, b=10),
        dragmode='zoom',
        xaxis=dict(
            title='Date',
            type='date',
            rangeslider=dict(visible=True),
            hoverformat='%b %d, %Y'
        ),
        yaxis=dict(
            title='Value ($)',
            tickprefix='$',
            tickformat=',.0f',
            automargin=True,
            rangemode='tozero'
        )
    )
    st.plotly_chart(fig1, use_container_width=True, config={'scrollZoom': False})

    st.subheader("Withdrawals Over Time")

    init_withdrawal = float(results_df['Withdrawal'].iloc[0])

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=results_df['Date'], y=results_df['Withdrawal'],
        mode='lines', name='Withdrawal', line=dict(color='#9467bd'),
        hovertemplate='<b>%{fullData.name}</b>: $%{y:,.2f}<extra></extra>'
    ))
    fig2.add_trace(go.Scatter(
        x=results_df['Date'],
        y=results_df['Fixed_WR_Withdrawal'] if 'Fixed_WR_Withdrawal' in results_df.columns else [init_withdrawal] * len(results_df),
        mode='lines',
        name='Initial Withdrawal',
        line=dict(color='#7f7f7f', dash='dash'),
        hovertemplate='<b>%{fullData.name}</b>: $%{y:,.2f}<extra></extra>'
    ))
    fig2.update_layout(
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        margin=dict(l=10, r=10, t=10, b=10),
        dragmode='zoom',
        xaxis=dict(
            title='Date',
            type='date',
            rangeslider=dict(visible=True),
            hoverformat='%b %d, %Y'
        ),
        yaxis=dict(
            title='Withdrawal ($/month)',
            tickprefix='$',
            tickformat=',.0f',
            automargin=True,
            rangemode='tozero'
        )
    )
    st.plotly_chart(fig2, use_container_width=True, config={'scrollZoom': False})

    # Totals summary under the withdrawals chart
    total_fixed = float(results_df['Fixed_WR_Withdrawal'].sum()) if 'Fixed_WR_Withdrawal' in results_df.columns else float(init_withdrawal) * len(results_df)
    total_guardrails = float(results_df['Withdrawal'].sum())
    diff_ratio = (total_guardrails - total_fixed) / total_fixed if total_fixed else 0.0

    st.markdown(f"Total Withdrawal (Fixed): ${total_fixed:,.0f}")
    st.markdown(f"Total Withdrawal (Using Guardrails): ${total_guardrails:,.0f}")
    st.markdown(f"Difference: {diff_ratio:+.0%}")

elif not is_guidance:
    if 'results_df' in st.session_state:
        results_df = st.session_state['results_df']

        if st.session_state.get('dirty'):
            render_dirty_banner()

        st.subheader("Simulation Mode")
        st.subheader("Portfolio Value vs Guardrails")

        show_guardrail_hits = st.checkbox(
            "Show guardrail hit markers",
            value=True,
            help="Toggle vertical dotted lines at guardrail hits.",
            key="show_guardrail_hits"
        )

        fig1 = go.Figure()
        # Add fixed-withdrawal portfolio value first so it renders behind others
        if 'Fixed_WR_Value' in results_df.columns:
            fig1.add_trace(go.Scatter(
                x=results_df['Date'], y=results_df['Fixed_WR_Value'],
                mode='lines', name='Value w/Fixed WR', line=dict(color='#7f7f7f'),
                opacity=0.6,
                hovertemplate='<b>%{fullData.name}</b>: $%{y:,.2f}<extra></extra>'
            ))
        # Draw guardrails first (muted) so portfolio is visually on top
        fig1.add_trace(go.Scatter(
            x=results_df['Date'], y=results_df['Upper_Guardrail'],
            mode='lines', name='Upper Guardrail', line=dict(color='#2ca02c'), opacity=0.45,
            hovertemplate='<b>%{fullData.name}</b>: $%{y:,.2f}<extra></extra>'
        ))
        fig1.add_trace(go.Scatter(
            x=results_df['Date'], y=results_df['Lower_Guardrail'],
            mode='lines', name='Lower Guardrail', line=dict(color='#d62728'), opacity=0.45,
            hovertemplate='<b>%{fullData.name}</b>: $%{y:,.2f}<extra></extra>'
        ))
        fig1.add_trace(go.Scatter(
            x=results_df['Date'], y=results_df['Portfolio_Value'],
            mode='lines', name='Portfolio Value', line=dict(color='#1f77b4'),
            hovertemplate='<b>%{fullData.name}</b>: $%{y:,.2f}<extra></extra>'
        ))
        # Add thin vertical dotted lines at guardrail hits (drawn behind data)
        shapes = []
        if show_guardrail_hits:
            for d in results_df.loc[results_df['Guardrail_Hit'] == 'UPPER', 'Date']:
                shapes.append(dict(
                    type='line', xref='x', yref='paper',
                    x0=d, x1=d, y0=0, y1=1,
                    line=dict(color='#2ca02c', width=1, dash='dot'),
                    layer='below'
                ))
            for d in results_df.loc[results_df['Guardrail_Hit'] == 'LOWER', 'Date']:
                shapes.append(dict(
                    type='line', xref='x', yref='paper',
                    x0=d, x1=d, y0=0, y1=1,
                    line=dict(color='#d62728', width=1, dash='dot'),
                    layer='below'
                ))

        fig1.update_layout(
            shapes=shapes,
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0, traceorder='reversed'),
            margin=dict(l=10, r=10, t=10, b=10),
            dragmode='zoom',
            xaxis=dict(
                title='Date',
                type='date',
                rangeslider=dict(visible=True),
                hoverformat='%b %d, %Y'
            ),
            yaxis=dict(
                title='Value ($)',
                tickprefix='$',
                tickformat=',.0f',
                automargin=True,
                rangemode='tozero'
            )
        )
        st.plotly_chart(fig1, use_container_width=True, config={'scrollZoom': False})

        st.subheader("Withdrawals Over Time")

        init_withdrawal = float(results_df['Withdrawal'].iloc[0])

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=results_df['Date'], y=results_df['Withdrawal'],
            mode='lines', name='Withdrawal', line=dict(color='#9467bd'),
            hovertemplate='<b>%{fullData.name}</b>: $%{y:,.2f}<extra></extra>'
        ))
        fig2.add_trace(go.Scatter(
            x=results_df['Date'],
            y=results_df['Fixed_WR_Withdrawal'] if 'Fixed_WR_Withdrawal' in results_df.columns else [init_withdrawal] * len(results_df),
            mode='lines',
            name='Initial Withdrawal',
            line=dict(color='#7f7f7f', dash='dash'),
            hovertemplate='<b>%{fullData.name}</b>: $%{y:,.2f}<extra></extra>'
        ))
        fig2.update_layout(
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
            margin=dict(l=10, r=10, t=10, b=10),
            dragmode='zoom',
            xaxis=dict(
                title='Date',
                type='date',
                rangeslider=dict(visible=True),
                hoverformat='%b %d, %Y'
            ),
            yaxis=dict(
                title='Withdrawal ($/month)',
                tickprefix='$',
                tickformat=',.0f',
                automargin=True,
                rangemode='tozero'
            )
        )
        st.plotly_chart(fig2, use_container_width=True, config={'scrollZoom': False})

        # Totals summary under the withdrawals chart
        total_fixed = float(results_df['Fixed_WR_Withdrawal'].sum()) if 'Fixed_WR_Withdrawal' in results_df.columns else float(init_withdrawal) * len(results_df)
        total_guardrails = float(results_df['Withdrawal'].sum())
        diff_ratio = (total_guardrails - total_fixed) / total_fixed if total_fixed else 0.0

        st.markdown(f"Total Withdrawal (Fixed): ${total_fixed:,.0f}")
        st.markdown(f"Total Withdrawal (Using Guardrails): ${total_guardrails:,.0f}")
        st.markdown(f"Difference: {diff_ratio:+.0%}")
    else:
        st.subheader("Simulation Mode")
        st.markdown("Use this mode to simulate running a guardrail-based retirement withdrawal strategy during a historical period.\n\n"
                    "For more information, see the [official documentation](https://github.com/rogercost/fire-guardrails/blob/main/README.md).")
        st.info("Adjust parameters in the sidebar and click 'Run Simulation' to start.")
