# Guardrails Withdrawal Strategy Toolkit

This repository contains an implementation of a novel guardrails-based dynamic portfolio withdrawal strategy.

The app can be found here: https://fire-guardrails.streamlit.app/

The premise is that by by adjusting spending in response to market conditions as we move through retirement, we can not 
only increase our chances of success, but spend more while doing it.

<a href="https://fire-guardrails.streamlit.app/~/+/?config=H4sIAMn6YmkC_22R0W7DIAxFf2XiOY1Ip65t_qFP056mCVngNGwEMgOtqmn_PpMo6iKVNzi-1_blRwzBoGjFqx2yg2SDfzqVl0rEBJSUgVRwczzKjTxsZMOEMFnCAT3jTJNIDcGnPor2-UVWAjy4W7RRrT0O-4YNZg_rbbLg1AVcZtjI6dSy9A36S406iVbW-10l2OOMScWsNcaoaHKT9fFuMjV3NxVH9Mb6M49x2E5mS8VCVLggkTUGvWg7cBErkccRSZ0zkCGwbmnEQxUHF66Pqay3u0UM5jPHNCXSEeiSyEr-kMu64ajuJPWEsQ_OFCR3K9YRfmf0-sY5nuZtyxctW2kYVRhnW_HmnR1sQvO_onMh0OOazvrlI9Sc9TRAJTTEnnVXXvb9g6-BrYqeq1fo9w8d2-uqRgIAAA">
  <img
    width="1422"
    height="945"
    alt="image"
    src="https://github.com/user-attachments/assets/e14cf616-40b3-4186-855e-b3ba765908b2"
  />
</a>

**All dollar amounts shown are in real dollars, net of inflation.**

## Disclaimer

This app is meant for theoretical experimentation purposes only. It is not financial advice and should not be used to provide financial advice. For guidance suitable for your own unique situation, needs and goals, please consult a fiduciary financial advisor.

## Background Information

* This [Kitces blog post](https://www.kitces.com/blog/risk-based-monte-carlo-probability-of-success-guardrails-retirement-distribution-hatchet/) lays out the conceptual framework. (It uses a Monte Carlo approach, where ours is a historical simulation.)
* The [Everyone Adjusts Toolkit](https://openpath.financial/guardrails/) created by Aubrey Williams of Open Path Financial,LLC contains a YouTube tutorial on how the strategy works, as well as a link to get more information.

## Assumptions

This tool is a strictly theoretical, academic utility, not a personalized financial planning tool (see Disclaimer above). It is not meant to realistically replicate the decision making process that an investor in the past would have undergone - instead, it is meant to provide a view on statistical characteristics of observed past market behavior, which may or may not be similar to the way the market behaves in the future.

Therefore it is important to keep in mind that running a historical period in the tool does not attempt to replicate what an investor back then should have done or would have been able to do. This is for two main reasons:
* The success rate simulation includes all periods up until the present. This means the "past investor" has a crystal ball into the future to see the statistical characteristics of how the market will behave.
* The computation does not correct for investing inefficiencies like fees, trading costs or rebalancing drift. This means the "past investor" has access to low-fee passive index funds, which did not exist before the 1970's.

The historical visualizations provided by the tool are thus better imagined as what would happen today if the market - in a totally unknowable future - happens to follow the exact same trajectory as it did in the past, rather than what an actual investor could have achieved during those historical periods.

## Data Sources

* This project uses Robert Shiller's historical market dataset found here: https://shillerdata.com/

## Quick Start - Streamlit App

You can run the Streamlit app locally on your PC. 

Ensure `uv` is installed: https://docs.astral.sh/uv/getting-started/installation/

Then start the app:
```
uv run streamlit run streamlit_app.py
```

It will redirect you to http://localhost:8501/ where you can interact with the UI.

## Reporting Bugs and Issues

This is a work in progress. Feel free to report issues or bugs, ask questions about the results, or suggest new features, 
[here](https://github.com/rogercost/fire-guardrails/issues).

When asking about or sharing a specific setup, find the "Shareable link to this run" link at the bottom of the page, right-click and copy it. This will help others reproduce your exact test case for debugging or exploration.

<img width="322" height="81" alt="image" src="https://github.com/user-attachments/assets/c07437c4-fbe2-41e0-a61e-e9cc448b9db0" />

Contributions are also welcome, please feel free to open a pull request.
