import os
import time
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd
import requests

from app_settings import Settings


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
def get_cached_shiller_df(session_state):
    """Return Shiller data from session cache, loading if necessary."""
    shiller_df = session_state.get('shiller_df')
    if shiller_df is None:
        shiller_df = load_shiller_data()
        session_state['shiller_df'] = shiller_df
    return shiller_df


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


def cashflow_schedule_for_window(cashflows, window_length, start_offset=0):
    """Build a per-month cashflow schedule for a retirement window."""
    schedule = np.zeros(int(window_length), dtype=np.float64)
    if not cashflows or window_length <= 0:
        return schedule

    for flow in cashflows:
        try:
            start = int(flow.get("start_month", 0))
            end = int(flow.get("end_month", -1))
            amount = float(flow.get("amount", 0.0))
        except (AttributeError, TypeError, ValueError):
            continue

        if end < start:
            continue

        adj_start = max(start - int(start_offset), 0)
        adj_end = end - int(start_offset)

        if adj_end < 0 or adj_start >= window_length:
            continue

        adj_end = min(adj_end, int(window_length) - 1)
        if adj_end < adj_start:
            continue

        schedule[adj_start:adj_end + 1] += amount

    return schedule


def compute_portfolio_returns(stock_prices, bond_prices, stock_pct):
    """Compute blended portfolio returns from stock and bond price series."""
    stock_returns = np.ones(len(stock_prices))
    stock_returns[1:] = stock_prices[1:] / stock_prices[:-1]

    bond_returns = np.ones(len(bond_prices))
    bond_returns[1:] = bond_prices[1:] / bond_prices[:-1]

    return float(stock_pct) * stock_returns + (1 - float(stock_pct)) * bond_returns


@nb.jit(nopython=True)
def test_all_periods(portfolio_returns, num_months, initial_value, monthly_spending, monthly_cashflows, final_value_target):
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
            withdrawal = monthly_spending - monthly_cashflows[i]
            if withdrawal < 0.0:
                withdrawal = 0.0
            value = value * portfolio_returns[start_idx + i] - withdrawal
            if value <= 0:
                break

        # Success requires: didn't deplete AND final value >= target
        if value > 0 and value >= final_value_target:
            successes += 1

    return successes / num_periods


def calculate_success_rate(df, withdrawal_rate, num_months, stock_pct=0.75,
                           analysis_start_date='1871-01-01', analysis_end_date=None,
                           initial_value=1_000_000, monthly_cashflows=None,
                           final_value_target=0.0):
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
    portfolio_returns = compute_portfolio_returns(stock_prices, bond_prices, stock_pct)
    monthly_spending = initial_value * withdrawal_rate / 12

    if monthly_cashflows is None:
        monthly_cashflows = np.zeros(num_months, dtype=np.float64)
    else:
        monthly_cashflows = np.asarray(monthly_cashflows, dtype=np.float64)
        if len(monthly_cashflows) != num_months:
            raise ValueError("monthly_cashflows length must match num_months")

    # Call the compiled function
    return test_all_periods(portfolio_returns, num_months, initial_value, monthly_spending, monthly_cashflows, final_value_target)


def get_spending_rate_for_fixed_success_rate(df, desired_success_rate, num_months,
                                             analysis_start_date='1871-01-01',
                                             analysis_end_date=None,
                                             initial_value=1_000_000, stock_pct=0.75,
                                             tolerance=0.001, max_iterations=50,
                                             verbose=False,
                                             cashflows=None,
                                             cashflow_start_offset=0,
                                             cashflow_schedule=None,
                                             final_value_target=0.0):
    """
    Compute the annual spending rate such that a historical simulation over periods of the desired length
    yields the desired success rate.

    Parameters
    ----------
    df : pd.DataFrame
        The main dataframe with market data
    desired_success_rate : float
        The target chance of not depleting the portfolio (e.g., 0.90 for 90%),
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
        - 'spending_rate': The annual spending rate that achieves the target success rate
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

    if cashflow_schedule is None:
        monthly_cashflows = cashflow_schedule_for_window(cashflows, num_months, cashflow_start_offset)
    else:
        monthly_cashflows = np.asarray(cashflow_schedule, dtype=np.float64)
        if len(monthly_cashflows) != num_months:
            raise ValueError("cashflow_schedule length must match num_months")

    # Binary search for the spending rate
    low_rate = 0  # 0.0% annual spending rate
    high_rate = 0.20  # 20% annual spending rate

    iteration = 0
    best_rate = None
    best_success_rate = None

    if verbose:
        print(f"Searching for spending rate with {desired_success_rate:.1%} success rate...")

    while iteration < max_iterations:
        mid_rate = (low_rate + high_rate) / 2

        current_success_rate = calculate_success_rate(
            df,
            mid_rate,
            num_months,
            stock_pct,
            analysis_start_date,
            analysis_end_date,
            initial_value,
            monthly_cashflows=monthly_cashflows,
            final_value_target=final_value_target,
        )

        # Check if we're within tolerance
        if abs(current_success_rate - desired_success_rate) <= tolerance:
            best_rate = mid_rate
            best_success_rate = current_success_rate
            if verbose:
                print(f"Converged! Found spending rate within tolerance.")
            break

        # Adjust search bounds
        # If success rate is too high, we can spend more
        if current_success_rate > desired_success_rate:
            low_rate = mid_rate
            if verbose:
                print(f"Success rate at {mid_rate} is too high ({current_success_rate:.3f} > {desired_success_rate:.3f}), "
                      f"increasing spending rate range to [{low_rate:.4f}, {high_rate:.4f}]")
        else:
            # If success rate is too low, we need to spend less
            high_rate = mid_rate
            if verbose:
                print(f"Success rate at {mid_rate} is too low ({current_success_rate:.3f} < {desired_success_rate:.3f}), "
                      f"decreasing spending rate range to [{low_rate:.4f}, {high_rate:.4f}]")

        best_rate = mid_rate
        best_success_rate = current_success_rate
        iteration += 1

    if iteration >= max_iterations:
        if verbose:
            print(f"Reached maximum iterations ({max_iterations}). Returning best found rate.")

    # Return results
    return {
        'spending_rate': best_rate,
        'actual_success_rate': best_success_rate,
        'num_simulations': num_paths,
        'iterations': iteration + 1
    }

def _compute_spending_rate_at_date(
    df,
    success_rate: float,
    months_remaining: int,
    analysis_start_date,
    current_date,
    current_portfolio_value: float,
    stock_pct: float,
    cashflows: list,
    cashflow_schedule_slice,
    final_value_target: float = 0.0,
) -> float:
    """Compute spending rate for a given success rate at a specific point in time."""
    result = get_spending_rate_for_fixed_success_rate(
        df=df,
        desired_success_rate=success_rate,
        num_months=months_remaining,
        analysis_start_date=analysis_start_date,
        analysis_end_date=current_date,
        initial_value=current_portfolio_value,
        stock_pct=stock_pct,
        cashflows=cashflows,
        cashflow_schedule=cashflow_schedule_slice,
        final_value_target=final_value_target,
    )
    return result['spending_rate']


def is_adjustment_month(ts: pd.Timestamp, adjustment_frequency: str) -> bool:
    """Determine whether the guardrail policy permits an adjustment in the given month."""

    month = int(ts.month)
    if adjustment_frequency == "Monthly":
        return True
    if adjustment_frequency == "Quarterly":
        return ((month - 1) % 3) == 0
    if adjustment_frequency == "Biannually":
        return month in (1, 7)
    if adjustment_frequency == "Annually":
        return month == 1
    # Fallback to monthly behaviour for unexpected values
    return True


def get_guardrail_withdrawals(
    df,
    settings: Settings,
    verbose=False,
    on_progress=None,
    on_status=None,
):
    """Creates a dataframe of withdrawals that follow an adaptive guardrail strategy.

    Parameters
    ----------
    df : pandas.DataFrame
        Historical dataset containing the Shiller return series.
    settings : Settings
        Snapshot of the control state selected in the UI.
    verbose : bool, optional
        Emit periodic status messages to stdout, by default False.
    on_progress : callable, optional
        Callback invoked with (current, total) as the simulation progresses.
    on_status : callable, optional
        Callback invoked with textual status updates.

    Returns
    -------
    pandas.DataFrame
        Dataframe with guardrail withdrawal results and diagnostics.
    """
    start_date = pd.to_datetime(settings.start_date)
    end_date = settings.retirement_end_date()
    if end_date is None:
        raise ValueError("Unable to determine retirement end date from settings.")

    # Filter to analysis period
    mask = (df["Date"] >= start_date) & (df["Date"] <= pd.to_datetime(end_date))
    subset = df[mask].copy().reset_index(drop=True)

    total_months = len(subset)
    cashflows = settings.cashflows_for_calculation()
    monthly_cashflows = cashflow_schedule_for_window(cashflows, total_months, 0)

    # Pre-compute portfolio returns for the entire historical period (for speed)
    all_stock_prices = df['Real Total Return Price'].values
    all_bond_prices = df['Real Total Bond Returns'].values
    portfolio_returns = compute_portfolio_returns(all_stock_prices, all_bond_prices, settings.stock_pct)

    # Initialize results storage
    results = []

    current_portfolio_value = float(settings.initial_value)
    previous_total_spending = float(settings.initial_monthly_spending)
    cap_multiplier = settings.spending_cap_multiplier
    floor_multiplier = settings.spending_floor_multiplier
    cap_amount = (
        previous_total_spending * float(cap_multiplier)
        if cap_multiplier is not None
        else None
    )
    floor_amount = (
        previous_total_spending * float(floor_multiplier)
        if floor_multiplier is not None
        else None
    )

    # Fixed-withdrawal shadow path initialization
    fixed_total_spending = previous_total_spending
    fixed_portfolio_value = float(settings.initial_value)

    guardrail_depleted = False
    fixed_depleted = False

    for i, row in subset.iterrows():
        current_date = row['Date']
        months_remaining = len(subset) - i
        is_first_month = i == 0
        adjustment_allowed = (not is_first_month) and is_adjustment_month(current_date, settings.adjustment_frequency)

        status_line = f"Processing {current_date.strftime('%Y-%m')}, portfolio=${current_portfolio_value:,.0f}, months_remaining={months_remaining}"
        if on_status is not None:
            on_status(status_line)
        if on_progress is not None:
            on_progress(i + 1, total_months)

        current_cashflow = float(monthly_cashflows[i]) if i < len(monthly_cashflows) else 0.0
        schedule_slice = monthly_cashflows[i:i + months_remaining]

        def get_sr(success_rate):
            return _compute_spending_rate_at_date(
                df, success_rate, months_remaining, settings.analysis_start_date,
                current_date, current_portfolio_value, settings.stock_pct,
                cashflows, schedule_slice,
                final_value_target=settings.final_value_target,
            )

        if not guardrail_depleted:

            # Calculate the upper and lower guardrail spending rates, based on the current spending and months remaining
            #
            upper_sr = get_sr(settings.upper_guardrail_success)
            lower_sr = get_sr(settings.lower_guardrail_success)

            # The guardrail portfolio values are the values that would result in the current spending amount
            # representing a probability of success equal to each guardrail's probability.
            #
            upper_guardrail_value = previous_total_spending / upper_sr * 12 if upper_sr > 0 else np.inf
            lower_guardrail_value = previous_total_spending / lower_sr * 12 if lower_sr > 0 else np.inf

            if verbose and i % 12 == 0:
                print(f"Processing {current_date.strftime('%Y-%m')}, portfolio=${current_portfolio_value:,.0f}, "
                      f"months_remaining={months_remaining}")

            if adjustment_allowed:
                # Step 3: Check if we hit guardrails
                hit_upper = current_portfolio_value >= upper_guardrail_value
                hit_lower = current_portfolio_value <= lower_guardrail_value

                # Step 4: Calculate proposed spending adjustment
                if hit_upper:
                    desired_success_rate = settings.upper_guardrail_success + settings.upper_adjustment_fraction * (
                        settings.target_success_rate - settings.upper_guardrail_success
                    )
                    new_sr = get_sr(desired_success_rate)
                    new_proposed_spending = current_portfolio_value * new_sr / 12
                    guardrail_hit = "UPPER"

                elif hit_lower:
                    desired_success_rate = settings.lower_guardrail_success + settings.lower_adjustment_fraction * (
                        settings.target_success_rate - settings.lower_guardrail_success
                    )
                    new_sr = get_sr(desired_success_rate)
                    new_proposed_spending = current_portfolio_value * new_sr / 12
                    guardrail_hit = "LOWER"

                else:
                    new_proposed_spending = previous_total_spending
                    guardrail_hit = "NONE"

                # Step 6: Only adjust if the new dollar amount differs from the previous dollar amount by more than the configured threshold.
                #
                bounded_spending = new_proposed_spending
                if cap_amount is not None:
                    bounded_spending = min(bounded_spending, cap_amount)
                if floor_amount is not None:
                    bounded_spending = max(bounded_spending, floor_amount)

                percent_change = (
                    abs(bounded_spending - previous_total_spending) / previous_total_spending
                    if previous_total_spending else 0.0
                )

                if percent_change > settings.adjustment_threshold:
                    # Make the adjustment
                    if len(results) == 0:
                        print(f"Setting spending_target to bounded_spending value of {bounded_spending}")
                    spending_target = bounded_spending
                    adjustment_made = True
                else:
                    # Keep previous spending (inflation adjusted)
                    spending_target = previous_total_spending
                    adjustment_made = False
            else:
                guardrail_hit = "SKIPPED"
                percent_change = 0.0
                spending_target = previous_total_spending
                adjustment_made = False
        else:
            upper_guardrail_value = 0.0
            lower_guardrail_value = 0.0
            spending_target = 0.0
            adjustment_made = False
            guardrail_hit = "DEPLETED"
            percent_change = 0.0
            adjustment_allowed = False

        withdrawal_amount = spending_target - current_cashflow
        if withdrawal_amount < 0.0:
            withdrawal_amount = 0.0
        # Fixed path withdrawal for this month
        fixed_actual_withdrawal = 0.0 if fixed_depleted else max(fixed_total_spending - current_cashflow, 0.0)

        # Store results
        results.append({
            'Date': current_date,
            'Withdrawal': withdrawal_amount,
            'Total_Spending': spending_target,
            'Net_Cashflow': current_cashflow,
            'Portfolio_Value': current_portfolio_value,
            'Fixed_SR_Value': fixed_portfolio_value,
            'Fixed_SR_Withdrawal': fixed_actual_withdrawal,
            'Fixed_SR_Total_Spending': fixed_total_spending,
            'Upper_Guardrail': upper_guardrail_value,
            'Lower_Guardrail': lower_guardrail_value,
            'Guardrail_Hit': guardrail_hit,
            'Percent_Change': percent_change,
            'Adjustment_Made': adjustment_made,
            'Adjustment_Allowed': adjustment_allowed
        })

        # Update portfolio values for next month
        # Apply withdrawals taken this month
        current_portfolio_value -= withdrawal_amount
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
        previous_total_spending = spending_target

    return pd.DataFrame(results)

def compute_guardrail_guidance_snapshot(
    df,
    asof_date,
    settings: Settings,
) -> dict:
    """Compute a single-point guidance snapshot (no historical loop).

    Parameters
    ----------
    df : pandas.DataFrame
        Historical dataset containing the Shiller return series.
    asof_date : datetime-like
        Date on which guidance is generated.
    settings : Settings
        Snapshot of the control state selected in the UI.

    Notes
    -----
    This mirrors the first-iteration logic in :func:`get_guardrail_withdrawals`:
      - Determine months_remaining based on [asof_date .. end_date]
      - Compute spending rates at target, upper, lower success rates using analysis_end_date=asof_date
      - Guardrail portfolio values are the PVs at which CURRENT spending would equal those spending rates
      - Hypothetical adjustments on guardrail hit are computed by moving partway back toward the target
        using the upper/lower adjustment fractions. Threshold gating is intentionally ignored.

    Adjustment frequency, thresholds, and spending bounds are not considered here.
    """
    asof = pd.to_datetime(asof_date)

    # Determine months remaining directly from configured duration
    num_months = int(settings.retirement_duration_months)
    if num_months <= 0:
        # Fallback to a standard 30-year horizon
        num_months = 360

    cashflows = settings.cashflows_for_calculation()
    monthly_cashflows = cashflow_schedule_for_window(cashflows, num_months, 0)
    current_cashflow_amount = float(monthly_cashflows[0]) if len(monthly_cashflows) > 0 else 0.0

    # Totals over the coming year for cashflows and withdrawals net of cashflows
    first_year_months = min(12, len(monthly_cashflows))
    annual_cashflow_total = float(np.sum(monthly_cashflows[:first_year_months])) if first_year_months else 0.0

    # Helper to compute spending rate for a given success rate at 'asof'
    def _sr(desired_success_rate: float):
        res = get_spending_rate_for_fixed_success_rate(
            df=df,
            desired_success_rate=float(desired_success_rate),
            num_months=num_months,
            analysis_start_date=settings.analysis_start_date,
            analysis_end_date=asof,
            initial_value=float(settings.initial_value),
            stock_pct=float(settings.stock_pct),
            cashflows=cashflows,
            cashflow_schedule=monthly_cashflows,
            final_value_target=float(settings.final_value_target),
        )
        return float(res["spending_rate"]) if res["spending_rate"] is not None else None

    target_sr = _sr(settings.target_success_rate)
    upper_sr = _sr(settings.upper_guardrail_success)
    lower_sr = _sr(settings.lower_guardrail_success)

    # Guardrail PVs where CURRENT spending equals the spending rate for upper/lower success rates (matches simulation logic)
    use_spending = (
        float(settings.initial_monthly_spending)
        if settings.initial_monthly_spending is not None
        else None
    )

    if upper_sr is not None and upper_sr > 0 and use_spending is not None:
        upper_guardrail_value = use_spending / upper_sr * 12.0
    else:
        upper_guardrail_value = np.inf

    if lower_sr is not None and lower_sr > 0 and use_spending is not None:
        lower_guardrail_value = use_spending / lower_sr * 12.0
    else:
        lower_guardrail_value = np.inf

    # Hypothetical adjustment targets if guardrails were hit (move back toward target)
    desired_upper_success_rate = settings.upper_guardrail_success + settings.upper_adjustment_fraction * (
        settings.target_success_rate - settings.upper_guardrail_success
    )
    desired_lower_success_rate = settings.lower_guardrail_success + settings.lower_adjustment_fraction * (
        settings.target_success_rate - settings.lower_guardrail_success
    )

    adj_upper_sr = _sr(desired_upper_success_rate)
    adj_lower_sr = _sr(desired_lower_success_rate)

    # At a guardrail hit, the simulation computes the new spending using the portfolio value at the hit.
    # Mirror that here by using the guardrail PVs rather than today's PV for the hypothetical adjustments
    # and apply the same cap/floor bounds that the simulation enforces.
    adj_upper_monthly = (
        float(upper_guardrail_value) * adj_upper_sr / 12.0
        if (adj_upper_sr is not None and np.isfinite(upper_guardrail_value))
        else None
    )
    adj_lower_monthly = (
        float(lower_guardrail_value) * adj_lower_sr / 12.0
        if (adj_lower_sr is not None and np.isfinite(lower_guardrail_value))
        else None
    )

    # Percent change relative to CURRENT spending
    def _pct_change(new_amt):
        if new_amt is None or float(settings.initial_monthly_spending) == 0.0:
            return None
        return (
            float(new_amt) - float(settings.initial_monthly_spending)
        ) / float(settings.initial_monthly_spending)

    # Implied spending rate from current spending and portfolio value
    implied_sr = (
        (float(settings.initial_monthly_spending) * 12.0 / float(settings.initial_value))
        if float(settings.initial_value) > 0
        else None
    )
    target_monthly_spending = (
        float(settings.initial_value) * target_sr / 12.0
        if target_sr is not None
        else None
    )
    target_monthly_withdrawal = None
    target_annual_withdrawal = None
    if target_monthly_spending is not None:
        target_monthly_withdrawal = max(target_monthly_spending - current_cashflow_amount, 0.0)
        if first_year_months:
            annual_withdrawals = [
                max(float(target_monthly_spending) - float(cf), 0.0)
                for cf in monthly_cashflows[:first_year_months]
            ]
            target_annual_withdrawal = float(np.sum(annual_withdrawals))

    # Calculate actual withdrawal rates (spending rate minus effect of cashflows)
    # Withdrawal rate = (monthly withdrawal * 12) / portfolio value
    implied_withdrawal_rate = None
    target_withdrawal_rate = None
    if float(settings.initial_value) > 0:
        current_withdrawal = max(float(settings.initial_monthly_spending) - current_cashflow_amount, 0.0)
        implied_withdrawal_rate = (current_withdrawal * 12.0) / float(settings.initial_value)
        if target_monthly_withdrawal is not None:
            target_withdrawal_rate = (target_monthly_withdrawal * 12.0) / float(settings.initial_value)

    return {
        "asof_date": asof,
        "months_remaining": num_months,
        "implied_spending_rate": implied_sr,
        "target_spending_rate": target_sr,
        "implied_withdrawal_rate": implied_withdrawal_rate,
        "target_withdrawal_rate": target_withdrawal_rate,
        "target_monthly_spending": target_monthly_spending,
        "target_monthly_withdrawal": target_monthly_withdrawal,
        "target_annual_withdrawal": target_annual_withdrawal,
        "upper_guardrail_value": upper_guardrail_value,
        "lower_guardrail_value": lower_guardrail_value,
        "upper_adjusted_monthly": adj_upper_monthly,
        "upper_adjustment_pct": _pct_change(adj_upper_monthly),
        "lower_adjusted_monthly": adj_lower_monthly,
        "lower_adjustment_pct": _pct_change(adj_lower_monthly),
        "current_cashflow": current_cashflow_amount,
        "annual_cashflow_total": annual_cashflow_total,
    }
