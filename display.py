import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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