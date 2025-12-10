import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from typing import Optional, Tuple
import utils


def _fmt_currency(value, escape_for_markdown: bool = False) -> str:
    """Format a currency value with optional Markdown escaping."""
    if value is None or (isinstance(value, float) and (pd.isna(value) or not np.isfinite(value))):
        return "N/A"
    prefix = "\\$" if escape_for_markdown else "$"
    if value < 0:
        return f"-{prefix}{abs(value):,.0f}"
    return f"{prefix}{value:,.0f}"


def update_isr_dynamic_label(isr_params: dict, cashflows: list):
    """
    Updates the dynamic label on the target success rate input that shows the corresponding initial spending rate.
    """
    shiller_df = utils.get_cached_shiller_df(st.session_state)

    res = utils.get_spending_rate_for_fixed_success_rate(
        df=shiller_df,
        desired_success_rate=isr_params['desired_success_rate'],
        num_months=isr_params['duration_months'],
        analysis_start_date=isr_params['analysis_start_date'],
        initial_value=isr_params['initial_value'],
        stock_pct=isr_params['stock_pct'],
        tolerance=0.001,
        max_iterations=50,
        verbose=False,
        cashflows=cashflows,
        final_value_target=isr_params.get('final_value_target', 0.0),
    )
    st.session_state['isr_value'] = float(res['spending_rate']) if res['spending_rate'] is not None else None
    st.session_state['isr_params'] = isr_params


def update_guardrail_dynamic_labels(gr_params: dict, cashflows: list):
    """
    Updates the dynamic labels on the upper and lower guardrail inputs that show the corresponding portfolio values.
    """
    shiller_df = utils.get_cached_shiller_df(st.session_state)

    # First-period spending used to determine portfolio values that align with the guardrails
    first_month_spending = float(gr_params.get('initial_spending', 0.0))
    final_value_target = gr_params.get('final_value_target', 0.0)

    # Compute spending rates at start of retirement using retirement start date as analysis end date
    upper_res = utils.get_spending_rate_for_fixed_success_rate(
        df=shiller_df,
        desired_success_rate=gr_params['upper_sr'],
        num_months=gr_params['duration_months'],
        analysis_start_date=gr_params['analysis_start_date'],
        initial_value=gr_params['initial_value'],
        stock_pct=gr_params['stock_pct'],
        tolerance=0.001,
        max_iterations=50,
        verbose=False,
        cashflows=cashflows,
        final_value_target=final_value_target,
    )

    lower_res = utils.get_spending_rate_for_fixed_success_rate(
        df=shiller_df,
        desired_success_rate=gr_params['lower_sr'],
        num_months=gr_params['duration_months'],
        analysis_start_date=gr_params['analysis_start_date'],
        initial_value=gr_params['initial_value'],
        stock_pct=gr_params['stock_pct'],
        tolerance=0.001,
        max_iterations=50,
        verbose=False,
        cashflows=cashflows,
        final_value_target=final_value_target,
    )

    upper_sr = float(upper_res['spending_rate']) if upper_res['spending_rate'] is not None else None
    lower_sr = float(lower_res['spending_rate']) if lower_res['spending_rate'] is not None else None

    upper_pv = first_month_spending / upper_sr * 12 if (upper_sr is not None and upper_sr > 0) else None
    lower_pv = first_month_spending / lower_sr * 12 if (lower_sr is not None and lower_sr > 0) else None

    st.session_state[
        'upper_label_suffix'] = f" (Initial PV: ${upper_pv:,.0f})" if upper_pv is not None else " (Initial PV: N/A)"
    st.session_state[
        'lower_label_suffix'] = f" (Initial PV: ${lower_pv:,.0f})" if lower_pv is not None else " (Initial PV: N/A)"

    upper_color = None
    lower_color = None

    try:
        initial_value = float(gr_params['initial_value'])
    except (KeyError, TypeError, ValueError):
        initial_value = None

    if initial_value is not None:
        if upper_pv is not None and upper_pv < initial_value:
            upper_color = "#d62728"
        if lower_pv is not None and lower_pv > initial_value:
            lower_color = "#d62728"

    st.session_state['upper_label_color'] = upper_color
    st.session_state['lower_label_color'] = lower_color
    st.session_state['guardrail_params'] = gr_params


def update_initial_spending_label(initial_spending: float,
                                  initial_value: float,
                                  auto_spending: Optional[float],
                                  overridden: bool) -> None:
    """Compute the dynamic label text and color for the Initial Monthly Spending control."""

    spending_rate = None
    if initial_value and initial_value > 0:
        spending_rate = (initial_spending * 12.0) / float(initial_value)

    if spending_rate is not None and np.isfinite(spending_rate):
        label_text = f"Initial Monthly Spending ({spending_rate * 100:.2f}% SR)"
    else:
        label_text = "Initial Monthly Spending (SR: N/A)"

    label_color = None
    if overridden and auto_spending is not None:
        if np.isclose(initial_spending, auto_spending, rtol=0.0, atol=0.5):
            label_color = None
        elif initial_spending > auto_spending:
            label_color = "#d62728"
        else:
            label_color = "#2ca02c"

    st.session_state['initial_spending_label_text'] = label_text
    st.session_state['initial_spending_label_color'] = label_color


def render_simulation_results(results_df: pd.DataFrame) -> None:
    """
    Renders charts and summaries for simulation results.
    """
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

    initial_total_spending = None
    total_spending_customdata = None
    if 'Total_Spending' in results_df.columns and not results_df.empty:
        initial_total_spending = float(results_df['Total_Spending'].iloc[0])
        total_spending_diff = results_df['Total_Spending'].astype(float) - initial_total_spending

        # Apply the formatting function to create a pre-formatted string for customdata
        formatted_total_spending_diff = total_spending_diff.apply(_fmt_currency)

        percent_denominator = initial_total_spending if initial_total_spending != 0 else np.nan
        total_spending_pct_diff = total_spending_diff / percent_denominator
        total_spending_customdata = np.column_stack([
            formatted_total_spending_diff,  # Use the pre-formatted string here as customdata[0]
            total_spending_pct_diff
        ])

    if 'Fixed_SR_Value' in results_df.columns:
        fig.add_trace(
            go.Scatter(
                x=results_df['Date'],
                y=results_df['Fixed_SR_Value'],
                mode='lines',
                name='Value w/Fixed SR',
                line=dict(color='#7f7f7f'),
                opacity=0.6,
                hovertemplate='<b>%{fullData.name}</b>: $%{y:,.0f}<extra></extra>'
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
            hovertemplate='<b>%{fullData.name}</b>: $%{y:,.0f}<extra></extra>'
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
            hovertemplate='<b>%{fullData.name}</b>: $%{y:,.0f}<extra></extra>'
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
            hovertemplate='<b>%{fullData.name}</b>: $%{y:,.0f}<extra></extra>'
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
            hovertemplate='<b>%{fullData.name}</b>: $%{y:,.0f}<extra></extra>'
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
                hovertemplate='<b>%{fullData.name}</b>: $%{y:,.0f}<extra></extra>'
            ),
            row=2,
            col=1
        )
    if 'Total_Spending' in results_df.columns:
        total_spending_hovertemplate = '<b>%{fullData.name}</b>: $%{y:,.0f}'
        if total_spending_customdata is not None:
            total_spending_hovertemplate += '<br>Difference: %{customdata[0]}'  # Use pre-formatted string directly
            if initial_total_spending not in (None, 0):
                total_spending_hovertemplate += '<br>% Difference: %{customdata[1]:+.1%}'
        total_spending_hovertemplate += '<extra></extra>'
        fig.add_trace(
            go.Scatter(
                x=results_df['Date'],
                y=results_df['Total_Spending'],
                mode='lines',
                name='Total Spending',
                line=dict(color='#bcbd22'),
                customdata=total_spending_customdata,
                hovertemplate=total_spending_hovertemplate
            ),
            row=2,
            col=1
        )
    fig.add_trace(
        go.Scatter(
            x=results_df['Date'],
            y=results_df['Fixed_SR_Withdrawal'] if 'Fixed_SR_Withdrawal' in results_df.columns else [init_withdrawal] * len(results_df),
            mode='lines',
            name='Initial Withdrawal',
            line=dict(color='#7f7f7f', dash='dash'),
            hovertemplate='<b>%{fullData.name}</b>: $%{y:,.0f}<extra></extra>'
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

    total_fixed = float(results_df['Fixed_SR_Withdrawal'].sum()) if 'Fixed_SR_Withdrawal' in results_df.columns else float(init_withdrawal) * len(results_df)
    total_guardrails = float(results_df['Withdrawal'].sum())
    total_cashflow = float(results_df['Net_Cashflow'].sum()) if 'Net_Cashflow' in results_df.columns else 0.0
    total_spending_guardrails = float(results_df['Total_Spending'].sum()) if 'Total_Spending' in results_df.columns else total_guardrails + total_cashflow
    total_spending_fixed = float(results_df['Fixed_SR_Total_Spending'].sum()) if 'Fixed_SR_Total_Spending' in results_df.columns else total_fixed + total_cashflow
    withdrawal_diff_ratio = (total_guardrails - total_fixed) / total_fixed if total_fixed else None
    spending_diff_ratio = (total_spending_guardrails - total_spending_fixed) / total_spending_fixed if total_spending_fixed else None

    if 'Total_Spending' in results_df.columns:
        spending_series = results_df['Total_Spending'].astype(float)
    else:
        spending_series = results_df['Withdrawal'].astype(float) if 'Withdrawal' in results_df.columns else pd.Series(dtype=float)
        if 'Net_Cashflow' in results_df.columns and not spending_series.empty:
            spending_series = spending_series + results_df['Net_Cashflow'].astype(float)

    start_spending = float(spending_series.iloc[0]) if not spending_series.empty else None
    min_spending = float(spending_series.min()) if not spending_series.empty else None
    max_spending = float(spending_series.max()) if not spending_series.empty else None

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

    min_streak = _longest_run(spending_series, min_spending) if not spending_series.empty else 0
    max_streak = _longest_run(spending_series, max_spending) if not spending_series.empty else 0

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
            "Metric": "Total Spending",
            "Fixed": _fmt_currency(total_spending_fixed),
            "Guardrails": _fmt_currency(total_spending_guardrails),
            "% Diff": _fmt_pct_diff(total_spending_guardrails, total_spending_fixed) if spending_diff_ratio is not None else "N/A",
        },
    ]

    st.table(pd.DataFrame(summary_rows).set_index("Metric"))

    spending_rows = [
        {
            "Metric": "Start Spending",
            "Monthly": _fmt_currency(start_spending),
            "% Diff": "—",
            "Duration (months)": "—",
        },
        {
            "Metric": "Min Spending",
            "Monthly": _fmt_currency(min_spending),
            "% Diff": _fmt_pct_diff(min_spending, start_spending),
            "Duration (months)": f"{min_streak}" if min_streak else "—",
        },
        {
            "Metric": "Max Spending",
            "Monthly": _fmt_currency(max_spending),
            "% Diff": _fmt_pct_diff(max_spending, start_spending),
            "Duration (months)": f"{max_streak}" if max_streak else "—",
        },
    ]

    st.table(pd.DataFrame(spending_rows).set_index("Metric"))


def render_guidance_results(snap: dict):
    """Summarize the guardrail guidance snapshot as advisor-friendly bullet points."""

    def fmt_money(x):
        return _fmt_currency(x, escape_for_markdown=True)

    def fmt_pct(p):
        return f"{p * 100:+.0f}%" if p is not None else "N/A"

    # Get spending rate (rate at which spending is calculated from portfolio)
    start_sr = st.session_state.get("isr_value", snap.get("target_spending_rate"))
    # Get actual withdrawal rate (true portfolio withdrawal rate after cashflows)
    target_wr = snap.get("target_withdrawal_rate")

    start_month = snap.get("target_monthly_spending")
    start_year = (start_month * 12.0) if start_month is not None else None
    start_net_withdrawal = snap.get("target_monthly_withdrawal")
    start_annual_withdrawal = snap.get("target_annual_withdrawal")

    upper_val = snap.get("upper_guardrail_value")
    lower_val = snap.get("lower_guardrail_value")

    up_adj_pct = snap.get("upper_adjustment_pct")
    up_adj_month = snap.get("upper_adjusted_monthly")
    up_adj_year = (up_adj_month * 12.0) if up_adj_month is not None else None

    low_adj_pct = snap.get("lower_adjustment_pct")
    low_adj_month = snap.get("lower_adjusted_monthly")
    low_adj_year = (low_adj_month * 12.0) if low_adj_month is not None else None

    cashflow_month0 = snap.get("current_cashflow")
    annual_cashflow_total = snap.get("annual_cashflow_total")

    st.subheader("Guidance Mode")
    st.markdown(
        "Use this mode to generate forward-looking guidance for a client who is retired today and in drawdown.\n\n"
        "All dollar amounts shown are in real (constant) dollars, net of inflation. For more details, see the [documentation](https://github.com/rogercost/fire-guardrails/blob/main/README.md).")

    # Notes for spending row shows spending rate
    spending_notes = (
        f"{start_sr * 100:.2f}% spending rate" if start_sr is not None else "Based on Initial Portfolio Value"
    )

    # Notes for withdrawal row shows actual withdrawal rate
    withdrawal_notes = (
        f"{target_wr * 100:.2f}% withdrawal rate" if target_wr is not None else "Net amount after monthly cashflows"
    )

    summary_rows = [
        {
            "Metric": "Target Spending",
            "Monthly": fmt_money(start_month),
            "Annual": fmt_money(start_year),
            "Notes": spending_notes,
        },
        {
            "Metric": "Portfolio Withdrawal After Cashflows",
            "Monthly": fmt_money(start_net_withdrawal),
            "Annual": fmt_money(start_annual_withdrawal),
            "Notes": withdrawal_notes,
        },
        {
            "Metric": "Current Monthly Cashflow",
            "Monthly": fmt_money(cashflow_month0),
            "Annual": fmt_money(annual_cashflow_total),
            "Notes": "First month and total for first year",
        },
    ]

    st.table(pd.DataFrame(summary_rows).set_index("Metric"))

    guardrail_rows = [
        {
            "Guardrail": "Upper",
            "Trigger (Portfolio Value)": fmt_money(upper_val),
            "Adjustment": fmt_pct(up_adj_pct),
            "New Spending (Monthly)": fmt_money(up_adj_month),
            "New Spending (Annual)": fmt_money(up_adj_year),
        },
        {
            "Guardrail": "Lower",
            "Trigger (Portfolio Value)": fmt_money(lower_val),
            "Adjustment": fmt_pct(low_adj_pct),
            "New Spending (Monthly)": fmt_money(low_adj_month),
            "New Spending (Annual)": fmt_money(low_adj_year),
        },
    ]

    st.table(pd.DataFrame(guardrail_rows).set_index("Guardrail"))
