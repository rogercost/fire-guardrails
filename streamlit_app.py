import streamlit as st
import pandas as pd
import datetime
import utils

st.set_page_config(layout="wide", page_title="Guardrail Withdrawal Simulator")

st.title("Guardrail-Based Withdrawal Strategy Simulator")
st.markdown("This application simulates a guardrail-based retirement withdrawal strategy based on historical market data.")

# Sidebar for inputs
st.sidebar.header("Simulation Parameters")

# Date Inputs
start_date = st.sidebar.date_input("Retirement Start Date", value=datetime.date(1968, 4, 1))
end_date = st.sidebar.date_input("Retirement End Date", value=datetime.date(2018, 3, 31))
analysis_start_date = st.sidebar.date_input("Historical Analysis Start Date", value=datetime.date(1871, 1, 1))

# Numeric Inputs
initial_value = st.sidebar.number_input("Initial Portfolio Value", value=1_000_000, min_value=100_000, step=100_000)
stock_pct = st.sidebar.slider("Stock Percentage", value=0.75, min_value=0.0, max_value=1.0, step=0.05)
target_success_rate = st.sidebar.slider("Target Success Rate", value=0.90, min_value=0.0, max_value=1.0, step=0.01)
upper_guardrail_success = st.sidebar.slider("Upper Guardrail Success Rate", value=1.00, min_value=0.0, max_value=1.0, step=0.01)
lower_guardrail_success = st.sidebar.slider("Lower Guardrail Success Rate", value=0.75, min_value=0.0, max_value=1.0, step=0.01)
upper_adjustment_fraction = st.sidebar.slider("Upper Adjustment Fraction", value=1.0, min_value=0.0, max_value=1.0, step=0.05)
lower_adjustment_fraction = st.sidebar.slider("Lower Adjustment Fraction", value=0.1, min_value=0.0, max_value=1.0, step=0.05)
adjustment_threshold = st.sidebar.slider("Adjustment Threshold (e.g., 0.05 for 5%)", value=0.05, min_value=0.0, max_value=0.2, step=0.01)

verbose = st.sidebar.checkbox("Verbose Output", value=False)

if st.sidebar.button("Run Simulation"):
    st.subheader("Running Simulation...")
    
    # Load data
    with st.spinner("Loading Shiller data..."):
        shiller_df = utils.load_shiller_data()
    st.success("Shiller data loaded.")

    # Run simulation
    with st.spinner("Calculating guardrail withdrawals..."):
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
            verbose=verbose
        )
    st.success("Simulation complete!")

    st.subheader("Simulation Results")
    st.dataframe(results_df)

    st.subheader("Portfolio Value and Withdrawals Over Time")
    
    # Placeholder for Highchart - using Streamlit's native line chart for now
    # For interactive Highcharts, you would typically use a library like `streamlit_highcharts`
    # or generate Highcharts JSON and use a custom Streamlit component.
    chart_data = results_df.set_index('Date')[['Portfolio_Value', 'Upper_Guardrail', 'Lower_Guardrail', 'Withdrawal']]
    st.line_chart(chart_data)

    st.info("Note: For a fully interactive Highchart with mouse-over details, further integration using a custom component or a dedicated Streamlit Highcharts library would be required.")

else:
    st.info("Adjust parameters in the sidebar and click 'Run Simulation' to start.")
