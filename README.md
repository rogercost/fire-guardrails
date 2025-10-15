# Guardrails Withdrawal Strategy Toolkit

This repository contains an implementation of a novel guardrails-based dynamic portfolio withdrawal strategy.

The premise is that by by adjusting spending in response to market conditions as we move through retirement, we can not 
only increase our chances of success, but spend more while doing it.

**All dollar amounts shown are in real dollars, net of inflation.**

## Background Information

* This [Kitces blog post](https://www.kitces.com/blog/risk-based-monte-carlo-probability-of-success-guardrails-retirement-distribution-hatchet/) lays out the conceptual framework. (It uses a Monte Carlo approach, where ours is a historical simulation.)
* The [Everyone Adjusts Toolkit](https://openpath.financial/guardrails/) from Open Path Financial contains a YouTube tutorial on how the strategy works, as well as a link to get more information.

## Data Sources

* This project uses Robert Shiller's historical market dataset found here: https://shillerdata.com/

## Reporting Bugs and Issues

This is a work in progress. Feel free to report issues or bugs, or suggest new features, 
[here](https://github.com/rogercost/fire-guardrails/issues).

Contributions are also welcome, please feel free to open a pull request.

## Quick Start - Streamlit App

You can run the Streamlit app locally on your PC. 

Ensure `uv` is installed: https://docs.astral.sh/uv/getting-started/installation/

Then start the app:
```
uv run streamlit run streamlit_app.py
```

It will redirect you to http://localhost:8501/ where you can interact with the UI.

## Guidance Mode (Client Assist)

The app includes a real-time Guidance Mode designed for client/adviser use. It computes a one-shot snapshot of:
- Starting withdrawal rate and amount (monthly/yearly) for the current portfolio value and remaining horizon
- Upper Guardrail portfolio value (where current spending would be considered “too low” by your upper success threshold), and the hypothetical increased spending if hit
- Lower Guardrail portfolio value (where current spending would be considered “too high” by your lower success threshold), and the hypothetical decreased spending if hit

How to use:
1. Use the Mode toggle at the top of the page and select “Guidance Mode”.
2. Enter your client’s Current Monthly Spending in the sidebar (defaults to $40,000).
3. Set Initial Portfolio Value (this represents today’s portfolio value in Guidance Mode), Stock Percentage, Target/Upper/Lower success rates, and Historical Analysis Start Date.
4. Guidance Mode will:
   - Fix “Retirement Start Date” to today (disabled)
   - Use “Retirement Duration (months)” as the remaining horizon from today
   - Disable “Adjustment Threshold” (internally treated as 0 for single-run guidance)
   - Hide the “Run Simulation” button (no historical loop is run)
   - Show a concise text output that auto-updates on any non-disabled control change

Output format example:
- Starting Withdrawal Rate: 4.35% ($4,236/month or $48,665/year)
- Upper Guardrail Portfolio Value: $1,234,567  ->  if hit, spending adjusts by -6% to $3,911/month or $43,210/year
- Lower Guardrail Portfolio Value: $888,777  ->  if hit, spending adjusts by +7% to $4,619/month or $56,789/year

Technical notes:
- Guidance Mode reuses the same success-rate solver as the simulation to compute target/upper/lower withdrawal rates for the configured Retirement Duration (months) from the as-of date.
- It uses the latest available month in the Shiller dataset (<= today) as the “as-of” date.
- “Initial Portfolio Value” in the sidebar is interpreted as today’s portfolio value in Guidance Mode.
- The hypothetical adjustments move part-way back toward the Target Success Rate using your Upper/Lower Adjustment Fractions (no threshold gating on a single snapshot).

## Python Notebook

The [Guardrails Notebook](./guardrails.ipynb) is a runnable Python notebook that lays out the approach in detail.
Feel free to experiment with it (use the Open in Colab button at the top; you'll need a Google account).
