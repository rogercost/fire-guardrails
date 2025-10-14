import datetime

import streamlit as st
import plotly.graph_objects as go

import utils

st.set_page_config(layout="wide", page_title="Guardrail Withdrawal Simulator")

title_ph = st.empty()
desc_ph = st.empty()
title_ph.title("Guardrail-Based Withdrawal Strategy Simulator")
desc_ph.markdown("This application simulates a guardrail-based retirement withdrawal strategy based on historical market data.")

# Sidebar for inputs
st.sidebar.header("Simulation Parameters")

# Date Inputs
start_date = st.sidebar.date_input(
    "Retirement Start Date",
    value=datetime.date(1968, 4, 1),
    min_value=datetime.date(1871, 1, 1),
    max_value=datetime.date.today(),
    help="First retirement month used for withdrawals. Simulations begin from this date."
)
end_date = st.sidebar.date_input(
    "Retirement End Date",
    value=datetime.date(2018, 3, 31),
    min_value=datetime.date(1871, 1, 1),
    max_value=datetime.date.today(),
    help="Final month to simulate withdrawals."
)
analysis_start_date = st.sidebar.date_input(
    "Historical Analysis Start Date",
    value=datetime.date(1871, 1, 1),
    min_value=datetime.date(1871, 1, 1),
    max_value=datetime.date.today(),
    help="Earliest date for historical data included to estimate success rates (Shiller data begins in 1871)."
)

# Numeric Inputs
initial_value = st.sidebar.number_input(
    "Initial Portfolio Value",
    value=1_000_000, min_value=100_000, step=100_000,
    help="Starting portfolio balance in dollars at retirement."
)
stock_pct = st.sidebar.slider(
    "Stock Percentage", value=0.75, min_value=0.0, max_value=1.0, step=0.05,
    help="Fraction of the portfolio allocated to stocks; remainder to bonds/cash."
)
target_success_rate = st.sidebar.slider(
    "Target Success Rate", value=0.90, min_value=0.0, max_value=1.0, step=0.01,
    help="Desired probability of sustaining withdrawals without depleting the portfolio across historical periods."
)
upper_guardrail_success = st.sidebar.slider(
    "Upper Guardrail Success Rate", value=1.00, min_value=0.0, max_value=1.0, step=0.01,
    help="If estimated success rises above this level, increase withdrawals (upper guardrail)."
)
lower_guardrail_success = st.sidebar.slider(
    "Lower Guardrail Success Rate", value=0.75, min_value=0.0, max_value=1.0, step=0.01,
    help="If estimated success falls below this level, decrease withdrawals (lower guardrail)."
)
upper_adjustment_fraction = st.sidebar.slider(
    "Upper Adjustment Fraction", value=1.0, min_value=0.0, max_value=1.0, step=0.05,
    help="Proportion to increase withdrawals when above the upper guardrail."
)
lower_adjustment_fraction = st.sidebar.slider(
    "Lower Adjustment Fraction", value=0.1, min_value=0.0, max_value=1.0, step=0.05,
    help="Proportion to decrease withdrawals when below the lower guardrail."
)
adjustment_threshold = st.sidebar.slider(
    "Adjustment Threshold (e.g., 0.05 for 5%)", value=0.05, min_value=0.0, max_value=0.2, step=0.01,
    help="Minimum absolute change in estimated success rate required before making an adjustment."
)

if st.sidebar.button(
    "Run Simulation",
    help="Fetch data and run the guardrail withdrawal simulation with the selected parameters."
):
    # Hide header/title block and use a single status line to save vertical space
    title_ph.empty()
    desc_ph.empty()

    status_ph = st.empty()
    status_ph.text("Loading Shiller data...")
    shiller_df = utils.load_shiller_data()
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
        end_date=end_date,
        analysis_start_date=analysis_start_date,
        initial_value=initial_value,
        stock_pct=stock_pct,
        target_success_rate=target_success_rate,
        upper_guardrail_success=upper_guardrail_success,
        lower_guardrail_success=lower_guardrail_success,
        upper_adjustment_fraction=upper_adjustment_fraction,
        lower_adjustment_fraction=lower_adjustment_fraction,
        adjustment_threshold=adjustment_threshold,
        verbose=True,
        on_progress=on_progress,
        on_status=on_status
    )

    # Remove status line entirely to place plots at the very top
    status_ph.empty()

    # Charts (table removed)
    st.subheader("Portfolio Value vs Guardrails")

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=results_df['Date'], y=results_df['Portfolio_Value'],
        mode='lines', name='Portfolio Value', line=dict(color='#1f77b4'),
        hovertemplate='<b>%{fullData.name}</b>: $%{y:,.2f}<extra></extra>'
    ))
    fig1.add_trace(go.Scatter(
        x=results_df['Date'], y=results_df['Upper_Guardrail'],
        mode='lines', name='Upper Guardrail', line=dict(color='#2ca02c'),
        hovertemplate='<b>%{fullData.name}</b>: $%{y:,.2f}<extra></extra>'
    ))
    fig1.add_trace(go.Scatter(
        x=results_df['Date'], y=results_df['Lower_Guardrail'],
        mode='lines', name='Lower Guardrail', line=dict(color='#d62728'),
        hovertemplate='<b>%{fullData.name}</b>: $%{y:,.2f}<extra></extra>'
    ))
    fig1.update_layout(
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
            title='Value ($)',
            tickprefix='$',
            tickformat=',.0f',
            automargin=True
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
        y=[init_withdrawal] * len(results_df),
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
            automargin=True
        )
    )
    st.plotly_chart(fig2, use_container_width=True, config={'scrollZoom': False})

else:
    st.info("Adjust parameters in the sidebar and click 'Run Simulation' to start.")
