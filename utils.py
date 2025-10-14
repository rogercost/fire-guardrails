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
                                  verbose=False,
                                  method='hybrid',
                                  use_exp_fit=True,
                                  wr_tolerance=None,
                                  initial_bounds=(0.0, 0.20),
                                  allow_expand=True,
                                  expand_factor=2.0,
                                  max_bound=1.0):
    """
    Compute the annual withdrawal rate such that a historical simulation over periods of the desired length
    yields the desired success rate.

    This version supports faster convergence via interpolation/extrapolation:
      - Secant steps on f(wr) = success_rate(wr) - target
      - Optional exponential fit of ln(1 - SR) ~ a + b*wr when 3+ points exist
      - Always maintains a bracket for safety; falls back to bisection when needed

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
    analysis_end_date : str | None, optional
        If set, limit the analysis to dates up to this point (historical-only).
    initial_value : float, optional
        Initial portfolio value (default 1,000,000)
    stock_pct : float, optional
        Percentage of portfolio in stocks (default 0.75)
    tolerance : float, optional
        Tolerance for success rate matching (default 0.001 = 0.1%)
    max_iterations : int, optional
        Maximum iterations for the solver (default 50)
    verbose : bool, optional
        Print progress (default False)
    method : {'binary','secant','hybrid'}, optional
        Which stepping strategy to use. 'hybrid' (default) uses exp-fit, then secant, then bisection fallback.
    use_exp_fit : bool, optional
        Try exponential fit ln(1 - SR) ~ a + b*wr when 3+ points exist (default True)
    wr_tolerance : float | None, optional
        Optional tolerance on the withdrawal rate bracket width for early stopping.
    initial_bounds : tuple(float,float), optional
        Initial [low, high] withdrawal-rate bounds (default (0.0, 0.20))
    allow_expand : bool, optional
        Allow expanding bounds if the target is not bracketed initially (default True)
    expand_factor : float, optional
        Factor to expand the interval by when bracketing (default 2.0)
    max_bound : float, optional
        Hard cap for the upper bound during expansion (default 1.0)

    Returns
    -------
    dict
        Dictionary containing:
        - 'withdrawal_rate': The annual withdrawal rate that achieves the target success rate
        - 'actual_success_rate': The actual success rate achieved
        - 'num_simulations': Number of simulation paths run (per evaluation)
        - 'iterations': Number of solver iterations performed
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

    # Convert analysis_start_date to datetime and compute data availability for num_paths info
    analysis_start = pd.to_datetime(analysis_start_date)
    df_filtered = df[df['Date'] >= analysis_start].copy()
    if analysis_end_date is not None:
        analysis_end = pd.to_datetime(analysis_end_date)
        df_filtered = df_filtered[df_filtered['Date'] <= analysis_end]

    max_end_idx = len(df_filtered) - num_months
    if max_end_idx <= 0:
        raise ValueError(f"Not enough data for {num_months} month simulations starting from {analysis_start_date}")
    num_paths = max_end_idx + 1

    # Bracket initialization
    low_rate, high_rate = initial_bounds
    low_rate = max(0.0, float(low_rate))
    high_rate = float(high_rate)
    if high_rate <= low_rate:
        raise ValueError("initial_bounds must satisfy high > low")

    # Cache evaluations: wr -> SR
    eval_cache = {}

    def eval_success_rate(wr: float) -> float:
        wr_clamped = max(0.0, min(float(wr), max_bound))
        if wr_clamped in eval_cache:
            return eval_cache[wr_clamped]
        sr = calculate_success_rate(
            df=df,
            withdrawal_rate=wr_clamped,
            num_months=num_months,
            stock_pct=stock_pct,
            analysis_start_date=analysis_start_date,
            analysis_end_date=analysis_end_date,
            initial_value=initial_value
        )
        eval_cache[wr_clamped] = sr
        return sr

    # Evaluate ends and attempt to ensure bracketing on f(wr) = SR(wr) - target
    sr_low = eval_success_rate(low_rate)
    sr_high = eval_success_rate(high_rate)
    f_low = sr_low - desired_success_rate
    f_high = sr_high - desired_success_rate

    if verbose:
        print(f"Initial bracket: wr in [{low_rate:.4f}, {high_rate:.4f}], "
              f"SR(low)={sr_low:.3f}, SR(high)={sr_high:.3f}, target={desired_success_rate:.3f}")

    # Expand bounds if not bracketed and allowed
    if allow_expand and f_low * f_high > 0:
        # If both above target, increase high until below or cap
        expand_iters = 0
        while f_low > 0 and f_high > 0 and high_rate < max_bound and expand_iters < 20:
            new_high = min(max_bound, high_rate * expand_factor if high_rate > 0 else initial_bounds[1] * expand_factor)
            if new_high == high_rate:  # cannot expand further
                break
            high_rate = new_high
            sr_high = eval_success_rate(high_rate)
            f_high = sr_high - desired_success_rate
            expand_iters += 1
            if verbose:
                print(f"Expanding high -> {high_rate:.4f}, SR={sr_high:.3f}")
        # If both below target, decrease low toward 0 until above or zero
        while f_low < 0 and f_high < 0 and low_rate > 0.0 and expand_iters < 40:
            new_low = max(0.0, low_rate / expand_factor)
            if new_low == low_rate:
                break
            low_rate = new_low
            sr_low = eval_success_rate(low_rate)
            f_low = sr_low - desired_success_rate
            expand_iters += 1
            if verbose:
                print(f"Expanding low -> {low_rate:.4f}, SR={sr_low:.3f}")

    # Points history for optional model fits
    points = {}  # wr -> SR
    points[low_rate] = sr_low
    points[high_rate] = sr_high

    def choose_next_wr() -> float:
        width = high_rate - low_rate
        mid = (low_rate + high_rate) / 2.0

        # 1) Exponential fit proposal if enabled and enough points
        wr_exp = None
        if method in ('hybrid', 'exp') and use_exp_fit and len(points) >= 3:
            wrs = np.array(sorted(points.keys()))
            srs = np.array([points[w] for w in wrs])
            # Guard: values must be strictly within (0,1) for log
            eps = 1e-12
            one_minus = np.clip(1.0 - srs, eps, 1.0 - eps)
            y = np.log(one_minus)
            X = np.vstack([np.ones_like(wrs), wrs]).T
            try:
                a, b = np.linalg.lstsq(X, y, rcond=None)[0]
                if np.isfinite(b) and abs(b) > 1e-12:
                    target_y = np.log(np.clip(1.0 - desired_success_rate, eps, 1.0 - eps))
                    wr_exp = (target_y - a) / b
            except Exception:
                wr_exp = None

        # 2) Secant proposal on bracket endpoints
        wr_secant = None
        if method in ('hybrid', 'secant') and (f_high - f_low) != 0.0:
            wr_secant = high_rate - f_high * (high_rate - low_rate) / (f_high - f_low)

        # 3) Midpoint fallback
        wr_mid = mid

        # Selection strategy: prefer in-bracket exp-fit, else in-bracket secant, else midpoint
        candidates = []
        if wr_exp is not None and low_rate < wr_exp < high_rate:
            candidates.append(('exp', wr_exp))
        if wr_secant is not None and low_rate < wr_secant < high_rate:
            candidates.append(('secant', wr_secant))

        if candidates:
            # Pick the candidate closer to mid (safer) to avoid extremes
            chosen = min(candidates, key=lambda kv: abs(kv[1] - mid))
            if verbose:
                print(f"Choosing {chosen[0]} step -> {chosen[1]:.6f}")
            return chosen[1]

        # Fallback: midpoint (bisection)
        if verbose:
            print(f"Falling back to bisection -> {wr_mid:.6f}")
        return wr_mid

    iterations = 0
    best_rate = None
    best_success_rate = None
    best_abs_err = float('inf')

    # Early convergence check on endpoints
    for wr_check, sr_check in [(low_rate, sr_low), (high_rate, sr_high)]:
        err = abs(sr_check - desired_success_rate)
        if err < best_abs_err:
            best_abs_err = err
            best_rate = wr_check
            best_success_rate = sr_check
    if best_abs_err <= tolerance:
        if verbose:
            print("Converged on an endpoint within tolerance.")
        return {
            'withdrawal_rate': best_rate,
            'actual_success_rate': best_success_rate,
            'num_simulations': num_paths,
            'iterations': iterations
        }

    if verbose:
        print(f"Searching for withdrawal rate with {desired_success_rate:.1%} success rate...")

    # Main solve loop
    while iterations < max_iterations:
        # Stop if bracket width small enough
        if wr_tolerance is not None and (high_rate - low_rate) <= wr_tolerance:
            # Choose the midpoint as final
            final_wr = (low_rate + high_rate) / 2.0
            final_sr = eval_success_rate(final_wr)
            if abs(final_sr - desired_success_rate) < best_abs_err:
                best_rate, best_success_rate, best_abs_err = final_wr, final_sr, abs(final_sr - desired_success_rate)
            if verbose:
                print(f"Bracket width {high_rate - low_rate:.6g} <= wr_tolerance {wr_tolerance}, stopping.")
            break

        wr_next = choose_next_wr()
        sr_next = eval_success_rate(wr_next)
        f_next = sr_next - desired_success_rate

        # Track best
        abs_err = abs(f_next)
        if abs_err < best_abs_err:
            best_abs_err = abs_err
            best_rate = wr_next
            best_success_rate = sr_next

        # Check convergence
        if abs_err <= tolerance:
            if verbose:
                print("Converged! Found withdrawal rate within tolerance.")
            best_rate = wr_next
            best_success_rate = sr_next
            iterations += 1
            break

        # Update bracket (SR decreases as WR increases)
        # If SR too high (> target), we can withdraw more -> move low up
        # If SR too low (< target), we must withdraw less -> move high down
        if f_next > 0:
            low_rate = wr_next
            sr_low = sr_next
            f_low = f_next
        else:
            high_rate = wr_next
            sr_high = sr_next
            f_high = f_next

        # Add to points for model fit
        points[wr_next] = sr_next

        iterations += 1

    if iterations >= max_iterations and verbose:
        print(f"Reached maximum iterations ({max_iterations}). Returning best found rate.")

    return {
        'withdrawal_rate': best_rate,
        'actual_success_rate': best_success_rate,
        'num_simulations': num_paths,
        'iterations': iterations
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

        # Calculate 3 withdrawal rates: the target, and the upper and lower guardrail, based on the current portfolio
        # value and months remaining
        #
        target_wr = get_withdrawal_rate(success_rate=target_success_rate)

        upper_wr = get_withdrawal_rate(success_rate=upper_guardrail_success)

        lower_wr = get_withdrawal_rate(success_rate=lower_guardrail_success)

        # The guardrail portfolio values are the values that would result in the **current** withdrawal amount
        # representing a probability of success equal to each guardrail's probability.
        #
        upper_guardrail_value = previous_monthly_spending / upper_wr * 12
        lower_guardrail_value = previous_monthly_spending / lower_wr * 12

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
        percent_change = abs(new_proposed_spending - previous_monthly_spending) / previous_monthly_spending

        if percent_change > adjustment_threshold:
            # Make the adjustment
            actual_monthly_spending = new_proposed_spending
            adjustment_made = True
        else:
            # Keep previous spending (inflation adjusted)
            actual_monthly_spending = previous_monthly_spending
            adjustment_made = False

        # Store results
        results.append({
            'Date': current_date,
            'Withdrawal': actual_monthly_spending,
            'Portfolio_Value': current_portfolio_value,
            'Target_WR': target_wr,
            'Upper_Guardrail': upper_guardrail_value,
            'Lower_Guardrail': lower_guardrail_value,
            'Guardrail_Hit': guardrail_hit,
            'Percent_Change': percent_change,
            'Adjustment_Made': adjustment_made
        })

        # Update portfolio value for next month
        # Apply withdrawal
        current_portfolio_value -= actual_monthly_spending

        # Apply market returns (if not at end)
        if i < len(subset) - 1:
            # Find the index in the full dataset
            full_idx = df[df['Date'] == current_date].index[0]
            month_return = portfolio_returns[full_idx + 1] if full_idx + 1 < len(portfolio_returns) else 1.0
            current_portfolio_value *= month_return

        # Update state
        previous_monthly_spending = actual_monthly_spending

        # Check for bankruptcy
        if current_portfolio_value <= 0:
            if verbose:
                print(f"Portfolio depleted at {current_date}")
            break

    return pd.DataFrame(results)
