import os
import time
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd
import requests


def parse_shiller_date(series):
    """
    Convert Shiller-style yyyy.mm strings/floats into proper datetimes.
    Examples:
      1950.01 -> 1950-01-01
      1950.1  -> 1950-10-01
      1950.11 -> 1950-11-01
    """
    s = series.astype(str).str.strip()

    def _norm(val):
        year, month_part = val.split(".", 1)
        # Special case: .1 means October (Excel dropped the zero)
        if month_part == "1":
            month = "10"
        else:
            month = month_part.zfill(2)
        return f"{year}-{month}-01"

    return pd.to_datetime(s.map(_norm), format="%Y-%m-%d")


def needs_update(path, days=30):
    """
    Checks if the supplied file exists and is less than 30 days old.
    """
    if not path.exists():
        return True
    age_days = (time.time() - path.stat().st_mtime) / (24 * 3600)
    return age_days > days


# Data loading function
def load_shiller_data():
    """
    Loads and preprocesses the Shiller data.
    (Stub - actual implementation will be copied from notebook)
    """
    local_path = Path(os.path.join("tempdir", "shillerdata.xls"))

    # This is the link found in https://shillerdata.com/
    url = ("https://img1.wsimg.com/blobby/go/e5e77e0b-59d1-44d9-ab25-4763ac982e53/downloads/"
           "9becfac9-1778-47a6-b40e-299d8c616706/ie_data.xls")

    # Download if missing or outdated
    if needs_update(local_path):
        local_path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(url)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(r.content)

    # Read multi-line headers (rows 4–7) and collapse them
    headers_raw = pd.read_excel(local_path, sheet_name="Data", skiprows=4, nrows=4, header=None)
    headers = (
        headers_raw.fillna("")
        .astype(str)
        .agg(" ".join)  # join rows into one string
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)  # clean spaces
    )

    # Read the actual data (starting row 8)
    df = pd.read_excel(local_path, sheet_name="Data", skiprows=8)
    df.columns = headers

    # Drop the last row which contains footnotes
    df = df.iloc[:-1].reset_index(drop=True)

    # TEMPORARY: Also drop the second to last row which as of 20251004 is a duplicate
    df = df.iloc[:-1].reset_index(drop=True)

    df["Date"] = parse_shiller_date(df["Date"])
    return df


@nb.jit(nopython=True)
def test_all_periods(portfolio_returns, num_months, initial_value, monthly_withdrawal):
    """
    Run all paths of a given length that exist in the supplied portfolio_reurns array,
    capturing the success or failure of each path, and returning the proportion of successes.

    We're running this in a tight loop so we'll Numba-compile it for ultra-fast simulation.
    """
    num_periods = len(portfolio_returns) - num_months + 1
    successes = 0

    for start_idx in range(num_periods):
        value = initial_value

        for i in range(num_months):
            value = value * portfolio_returns[start_idx + i] - monthly_withdrawal
            if value <= 0:
                break

        if value > 0:
            successes += 1

    return successes / num_periods


def calculate_success_rate(df, withdrawal_rate, num_months, stock_pct=0.75,
                           analysis_start_date='1871-01-01', analysis_end_date=None, initial_value=1_000_000):
    """
    Calculate the success rate for a given withdrawal rate.
    Numba-accelerated version. Should be 500x+ faster than brute force.
    First call will be slower due to compilation, subsequent calls will be blazing fast.
    That way we can use it iteratively in a guess-and-check loop.
    """
    # Prepare data
    analysis_start = pd.to_datetime(analysis_start_date)
    df_filtered = df[df['Date'] >= analysis_start]

    if analysis_end_date is not None:
        analysis_end = pd.to_datetime(analysis_end_date)
        df_filtered = df_filtered[df_filtered['Date'] <= analysis_end]

    # Calculate portfolio returns
    stock_prices = df_filtered['Real Total Return Price'].values
    bond_prices = df_filtered['Real Total Bond Returns'].values

    stock_returns = np.ones(len(stock_prices))
    stock_returns[1:] = stock_prices[1:] / stock_prices[:-1]

    bond_returns = np.ones(len(bond_prices))
    bond_returns[1:] = bond_prices[1:] / bond_prices[:-1]

    portfolio_returns = stock_pct * stock_returns + (1 - stock_pct) * bond_returns
    monthly_withdrawal = initial_value * withdrawal_rate / 12

    # Call the compiled function
    return test_all_periods(portfolio_returns, num_months, initial_value, monthly_withdrawal)


def get_wr_for_fixed_success_rate(df, desired_success_rate, num_months,
                                  analysis_start_date='1871-01-01',
                                  analysis_end_date=None,
                                  initial_value=1_000_000, stock_pct=0.75,
                                  tolerance=0.001, max_iterations=50,
                                  verbose=False):
    """
    Compute the annual withdrawal rate such that a historical simulation over periods of the desired length
    yields the desired success rate.

    Parameters
    ----------
    df : pd.DataFrame
        The main dataframe with market data
    desired_success_rate : float
        The target chance of underspending (e.g., 0.90 for 90%),
        the percent of simulation paths that should have ending portfolio values > 0.
    num_months : int
        The size of the time window of the historical simulation paths to run
        (corresponds to the remaining time in retirement).
    analysis_start_date : str, optional
        The start date from which we should begin running simulation paths,
        if we do not want to start at the very beginning.
    initial_value : float, optional
        Initial portfolio value (default 1,000,000)
    stock_pct : float, optional
        Percentage of portfolio in stocks (default 0.75)
    tolerance : float, optional
        Tolerance for success rate matching (default 0.001 = 0.1%)
    max_iterations : int, optional
        Maximum iterations for binary search (default 50)
    verbose : bool, optional
        Print progress (default False)

    Returns
    -------
    dict
        Dictionary containing:
        - 'withdrawal_rate': The annual withdrawal rate that achieves the target success rate
        - 'actual_success_rate': The actual success rate achieved
        - 'num_simulations': Number of simulation paths run
        - 'iterations': Number of binary search iterations performed
    """

    # Edge case: we can't goal seek if the algo is one ended
    if desired_success_rate > 1.0 - tolerance:
        if verbose:
            print(f"Edge case: reducing desired success rate from {desired_success_rate} to effective algo "
                  f"ceiling of {1.0 - tolerance}")
        desired_success_rate = 1.0 - tolerance

    if desired_success_rate < tolerance:
        if verbose:
            print(f"Edge case: increasing desired success rate from {desired_success_rate} to effective algo "
                  f"floor of {tolerance}")
        desired_success_rate = tolerance

    # Convert analysis_start_date to datetime
    analysis_start = pd.to_datetime(analysis_start_date)

    # Filter dataframe to only include dates from analysis_start_date onwards
    df_filtered = df[df['Date'] >= analysis_start].copy()

    # Optional: propose an end date. If we're running a historical simulation of this method, we cannot allow
    # access to future data.
    if analysis_end_date is not None:
        analysis_end = pd.to_datetime(analysis_end_date)
        df_filtered = df_filtered[df_filtered['Date'] <= analysis_end]

    # Get all possible starting dates for simulation periods
    max_end_idx = len(df_filtered) - num_months
    if max_end_idx <= 0:
        raise ValueError(f"Not enough data for {num_months} month simulations starting from {analysis_start_date}")

    num_paths = max_end_idx + 1

    # Binary search for the withdrawal rate
    low_rate = 0  # 0.0% annual withdrawal rate
    high_rate = 0.20  # 20% annual withdrawal rate

    iteration = 0
    best_rate = None
    best_success_rate = None

    if verbose:
        print(f"Searching for withdrawal rate with {desired_success_rate:.1%} success rate...")

    while iteration < max_iterations:
        mid_rate = (low_rate + high_rate) / 2

        current_success_rate = calculate_success_rate(df, mid_rate, num_months, stock_pct,
                                                      analysis_start_date, analysis_end_date, initial_value)

        # Check if we're within tolerance
        if abs(current_success_rate - desired_success_rate) <= tolerance:
            best_rate = mid_rate
            best_success_rate = current_success_rate
            if verbose:
                print(f"Converged! Found withdrawal rate within tolerance.")
            break

        # Adjust search bounds
        # If success rate is too high, we can withdraw more
        if current_success_rate > desired_success_rate:
            low_rate = mid_rate
            if verbose:
                print(f"Success rate at {mid_rate} is too high ({current_success_rate:.3f} > {desired_success_rate:.3f}), "
                      f"increasing withdrawal rate range to [{low_rate:.4f}, {high_rate:.4f}]")
        else:
            # If success rate is too low, we need to withdraw less
            high_rate = mid_rate
            if verbose:
                print(f"Success rate at {mid_rate} is too low ({current_success_rate:.3f} < {desired_success_rate:.3f}), "
                      f"decreasing withdrawal rate range to [{low_rate:.4f}, {high_rate:.4f}]")

        best_rate = mid_rate
        best_success_rate = current_success_rate
        iteration += 1

    if iteration >= max_iterations:
        if verbose:
            print(f"Reached maximum iterations ({max_iterations}). Returning best found rate.")

    # Return results
    return {
        'withdrawal_rate': best_rate,
        'actual_success_rate': best_success_rate,
        'num_simulations': num_paths,
        'iterations': iteration + 1
    }


def get_guardrail_withdrawals(df, start_date, end_date,
                              analysis_start_date='1871-01-01',
                              initial_value=1_000_000,
                              stock_pct=0.75,
                              target_success_rate=0.90,
                              upper_guardrail_success=1.00,
                              lower_guardrail_success=0.75,
                              upper_adjustment_fraction=1.0,
                              lower_adjustment_fraction=0.1,
                              adjustment_threshold=0.05,
                              verbose=False,
                              on_progress=None,
                              on_status=None):
    """
    Creates a dataframe of withdrawals that follow an adaptive guardrail strategy.

    The strategy:
    1. Each month, calculate the withdrawal rate that gives target_success_rate
    2. Calculate portfolio values that would make current spending have upper/lower success rates
    3. If portfolio hits guardrails, adjust spending
    4. Only implement adjustment if it exceeds threshold (to avoid constant small changes)

    Returns
    -------
    pd.DataFrame
        A dataframe with Date, Withdrawal, and diagnostic columns.
    """
    # Filter to analysis period
    mask = (df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))
    subset = df[mask].copy().reset_index(drop=True)
    
    total_months = len(subset)

    # Pre-compute portfolio returns for the entire historical period (for speed)
    all_stock_prices = df['Real Total Return Price'].values
    all_bond_prices = df['Real Total Bond Returns'].values

    stock_returns = np.ones(len(all_stock_prices))
    stock_returns[1:] = all_stock_prices[1:] / all_stock_prices[:-1]

    bond_returns = np.ones(len(all_bond_prices))
    bond_returns[1:] = all_bond_prices[1:] / all_bond_prices[:-1]

    portfolio_returns = stock_pct * stock_returns + (1 - stock_pct) * bond_returns

    # Initialize results storage
    results = []

    # Get the initial withdrawal rate based on the target success rate
    #
    initial_wr = get_wr_for_fixed_success_rate(df=df,
                                               desired_success_rate=target_success_rate,
                                               num_months=len(subset),
                                               analysis_start_date=analysis_start_date,
                                               analysis_end_date=start_date,  # We're starting this in the past!
                                               initial_value=initial_value,
                                               stock_pct=stock_pct,
                                               tolerance=0.001,
                                               max_iterations=50,
                                               verbose=False)['withdrawal_rate']

    if verbose:
        print(f"Initial withdrawal rate: {(initial_wr * 100):.2f}%")

    # State variables
    current_portfolio_value = initial_value
    previous_monthly_spending = initial_value * initial_wr / 12

    # Fixed-withdrawal shadow path initialization
    fixed_monthly_spending = previous_monthly_spending
    fixed_portfolio_value = initial_value

    guardrail_depleted = False
    fixed_depleted = False

    for i, row in subset.iterrows():
        current_date = row['Date']
        months_remaining = len(subset) - i

        status_line = f"Processing {current_date.strftime('%Y-%m')}, portfolio=${current_portfolio_value:,.0f}, months_remaining={months_remaining}"
        if on_status is not None:
            on_status(status_line)
        if on_progress is not None:
            on_progress(i + 1, total_months)

        def get_withdrawal_rate(success_rate):
            return get_wr_for_fixed_success_rate(df=df,
                                                 desired_success_rate=success_rate,
                                                 num_months=months_remaining,
                                                 analysis_start_date=analysis_start_date,
                                                 analysis_end_date=current_date,
                                                 initial_value=current_portfolio_value,
                                                 stock_pct=stock_pct,
                                                 tolerance=0.001,
                                                 max_iterations=50,
                                                 verbose=False)['withdrawal_rate']

        if not guardrail_depleted:
            # Calculate 3 withdrawal rates: the target, and the upper and lower guardrail, based on the current portfolio
            # value and months remaining
            #
            target_wr = get_withdrawal_rate(success_rate=target_success_rate)

            upper_wr = get_withdrawal_rate(success_rate=upper_guardrail_success)

            lower_wr = get_withdrawal_rate(success_rate=lower_guardrail_success)

            # The guardrail portfolio values are the values that would result in the current withdrawal amount
            # representing a probability of success equal to each guardrail's probability.
            #
            upper_guardrail_value = previous_monthly_spending / upper_wr * 12 if upper_wr > 0 else np.inf
            lower_guardrail_value = previous_monthly_spending / lower_wr * 12 if lower_wr > 0 else np.inf

            if verbose and i % 12 == 0:
                print(f"Processing {current_date.strftime('%Y-%m')}, portfolio=${current_portfolio_value:,.0f}, "
                      f"months_remaining={months_remaining}")

            # Step 3: Check if we hit guardrails
            hit_upper = current_portfolio_value >= upper_guardrail_value
            hit_lower = current_portfolio_value <= lower_guardrail_value

            # Step 4: Calculate proposed spending adjustment
            if hit_upper:
                desired_success_rate = upper_guardrail_success + upper_adjustment_fraction * (target_success_rate - upper_guardrail_success)
                new_wr = get_withdrawal_rate(success_rate=desired_success_rate)
                new_proposed_spending = current_portfolio_value * new_wr / 12
                guardrail_hit = "UPPER"

            elif hit_lower:
                desired_success_rate = lower_guardrail_success + lower_adjustment_fraction * (target_success_rate - lower_guardrail_success)
                new_wr = get_withdrawal_rate(success_rate=desired_success_rate)
                new_proposed_spending = current_portfolio_value * new_wr / 12
                guardrail_hit = "LOWER"

            else:
                new_proposed_spending = previous_monthly_spending
                guardrail_hit = "NONE"

            # Step 6: Only adjust if the new dollar amount differs from the previous dollar amount by more than the configured threshold.
            #
            percent_change = abs(new_proposed_spending - previous_monthly_spending) / previous_monthly_spending if previous_monthly_spending else 0.0

            if percent_change > adjustment_threshold:
                # Make the adjustment
                actual_monthly_spending = new_proposed_spending
                adjustment_made = True
            else:
                # Keep previous spending (inflation adjusted)
                actual_monthly_spending = previous_monthly_spending
                adjustment_made = False
        else:
            target_wr = np.nan
            upper_guardrail_value = 0.0
            lower_guardrail_value = 0.0
            actual_monthly_spending = 0.0
            adjustment_made = False
            guardrail_hit = "DEPLETED"
            percent_change = 0.0

        # Fixed path withdrawal for this month
        fixed_actual_withdrawal = 0.0 if fixed_depleted else fixed_monthly_spending

        # Store results
        results.append({
            'Date': current_date,
            'Withdrawal': actual_monthly_spending,
            'Portfolio_Value': current_portfolio_value,
            'Fixed_WR_Value': fixed_portfolio_value,
            'Fixed_WR_Withdrawal': fixed_actual_withdrawal,
            'Target_WR': target_wr,
            'Upper_Guardrail': upper_guardrail_value,
            'Lower_Guardrail': lower_guardrail_value,
            'Guardrail_Hit': guardrail_hit,
            'Percent_Change': percent_change,
            'Adjustment_Made': adjustment_made
        })

        # Update portfolio values for next month
        # Apply withdrawals taken this month
        current_portfolio_value -= actual_monthly_spending
        fixed_portfolio_value -= fixed_actual_withdrawal

        # Apply market returns (if not at end)
        if i < len(subset) - 1:
            # Find the index in the full dataset
            full_idx = df[df['Date'] == current_date].index[0]
            month_return = portfolio_returns[full_idx + 1] if full_idx + 1 < len(portfolio_returns) else 1.0
            current_portfolio_value *= month_return
            fixed_portfolio_value *= month_return

        # Floor at zero and mark depletion so future months remain at zero
        if current_portfolio_value <= 0:
            current_portfolio_value = 0.0
            guardrail_depleted = True
        if fixed_portfolio_value <= 0:
            fixed_portfolio_value = 0.0
            fixed_depleted = True

        # Update state
        previous_monthly_spending = actual_monthly_spending

    return pd.DataFrame(results)
