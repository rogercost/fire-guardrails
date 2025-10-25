# Guardrails Withdrawal Strategy Toolkit

This repository contains an implementation of a novel guardrails-based dynamic portfolio withdrawal strategy.

The app can be found here: https://fire-guardrails.streamlit.app/

The premise is that by by adjusting spending in response to market conditions as we move through retirement, we can not 
only increase our chances of success, but spend more while doing it.

**All dollar amounts shown are in real dollars, net of inflation.**

## Disclaimer

This app is meant for theoretical experimentation purposes only. It is not financial advice and should not be used to provide financial advice. For guidance suitable for your own unique situation, needs and goals, please consult a fiduciary financial advisor.

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

## Python Notebook

The [Guardrails Notebook](./guardrails.ipynb) is a runnable Python notebook that lays out the approach in detail.
Feel free to experiment with it (use the Open in Colab button at the top; you'll need a Google account).
