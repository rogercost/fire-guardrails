# fire-guardrails
An implementation of a novel guardrails-based dynamic portfolio withdrawal strategy for retirees.

The following detailed instructions outline the implementation of a Risk-Based Guardrails withdrawal strategy, emphasizing the use of Historical Analysis (specifically FireCalc, as detailed in the sources) and translating probability-of-success metrics into concrete dollar values for effective client communication.

---

# Risk-Based Guardrails Withdrawal Strategy Implementation Guide

**Intended Audience:** Financial Advisors (Fiduciary Standard Recommended).

**Core Principle:** This strategy recognizes that clients inevitably adjust their spending in retirement and provides a systematic, planned approach based on historical analysis and risk appetite, rather than relying on fixed withdrawal rates. The guardrails are defined around overall plan risk levels, not crude distribution rates, thus accounting for factors like Social Security, pensions, and non-portfolio cash flows.

## Phase I: Initial Guardrail Setup (The Start)

The initial setup requires defining client-specific parameters and calculating four key metrics: Initial Withdrawal Amount, Lower Guardrail Portfolio Value, Lower Guardrail Adjusted Spending, Upper Guardrail Portfolio Value, and Upper Guardrail Adjusted Spending.

### Step 1: Collect Client Model Parameters

Gather all necessary inputs, which incorporate client-specific nuances that traditional fixed withdrawal rates often overlook:

| Parameter | Input Detail | Source Reference |
| :--- | :--- | :--- |
| **Initial Portfolio Value** | Current liquid net worth used for retirement funding (e.g., $1,000,000). | |
| **Time Horizon (Years)** | The number of years the plan must last (e.g., 50 years). | |
| **Asset Allocation** | Percentage of the portfolio allocated to equities/stocks (e.g., 75%). The remainder is assumed to be fixed income. | |
| **Expense Ratio** | The weighted-average expense ratio for the investable assets (e.g., 0.03%). | |
| **Other Income** | Any guaranteed, non-portfolio income, such as Social Security (SS) or pensions. Must be entered as **annual amounts** and include the year/age they begin. | |
| **One-Time Flows** | Anticipated major inflows (inheritance, home sale) or outflows (home purchase) over the planning horizon. | |
| **Risk Targets (Chance of Underspending)** | Define the target probability of success levels for the plan: | |
| | **Initial Target Success Rate:** (e.g., 90%) | |
| | **Lower Guardrail Success Rate:** (e.g., 75%) | |
| | **Upper Guardrail Success Rate:** (e.g., 100%) | |

*Note: Aubrey Williams defines 100% Chance of Success as 100% Chance of Underspending—meaning you are guaranteed to end up with money left over—and encourages selecting an Initial Target lower than 100% to spend more.*

### Step 2: Determine Initial Withdrawal Amount

Calculate the starting Annual Withdrawal Amount, which corresponds to the **Initial Target Success Rate** (e.g., 90%) using the planning software (e.g., FireCalc):

1.  Enter all collected parameters into the planning software (Portfolio Value, Years, Income, Allocation).
2.  In the investigation tool, select **Spending Level** and input the **Initial Target Success Rate** (e.g., 90%).
3.  The resulting output is the **Initial Annual Withdrawal Amount** (Gross, before taxes).
    *   *Calculation:* Initial Monthly Withdrawal = Initial Annual Withdrawal Amount / 12.

### Step 3: Determine Lower Guardrail (LGR) Thresholds

The LGR identifies the portfolio value drop that triggers a spending reduction and calculates the corresponding new spending amount:

#### A. Calculate Lower Guardrail Portfolio Value (The Trigger)

1.  Use the **Initial Annual Withdrawal Amount** (from Step 2) as the Spending input in the software.
2.  In the investigation tool, select **Starting Portfolio Value** and input the **Lower Guardrail Success Rate** (e.g., 75%).
3.  The resulting output is the **Lower Guardrail Portfolio Value**. If the client’s portfolio drops to this value, an adjustment is required.

#### B. Calculate Lower Guardrail Adjusted Spending (The Adjustment)

1.  Set the Portfolio Value in the software to the **Lower Guardrail Portfolio Value** (from Step 3A).
2.  In the investigation tool, select **Spending Level** and input the **Initial Target Success Rate** (e.g., 90%).
    *   *Note: This step calculates the spending required to restore the risk level back to the initial, preferred target.*
3.  The resulting output is the **Adjusted Annual Spending Level**.
    *   *Calculation:* LGR Adjusted Monthly Withdrawal = Adjusted Annual Spending Level / 12.
    *   *Change Calculation:* Monthly Adjustment = Initial Monthly Withdrawal – LGR Adjusted Monthly Withdrawal (This confirms the dollar amount of the cut).

### Step 4: Determine Upper Guardrail (UGR) Thresholds

The UGR identifies the portfolio value increase that triggers a spending increase and calculates the corresponding new spending amount:

#### A. Calculate Upper Guardrail Portfolio Value (The Trigger)

1.  Restore the Initial Portfolio Value ($1,000,000 in the example) in the software.
2.  Use the **Initial Annual Withdrawal Amount** (from Step 2) as the Spending input.
3.  In the investigation tool, select **Starting Portfolio Value** and input the **Upper Guardrail Success Rate** (e.g., 100%).
4.  The resulting output is the **Upper Guardrail Portfolio Value**. If the client’s portfolio grows to this value, an adjustment is required.

#### B. Calculate Upper Guardrail Adjusted Spending (The Adjustment)

1.  Set the Portfolio Value in the software to the **Upper Guardrail Portfolio Value** (from Step 4A).
2.  In the investigation tool, select **Spending Level** and input the **Initial Target Success Rate** (e.g., 90%).
3.  The resulting output is the **Adjusted Annual Spending Level**.
    *   *Calculation:* UGR Adjusted Monthly Withdrawal = Adjusted Annual Spending Level / 12.
    *   *Change Calculation:* Monthly Increase = UGR Adjusted Monthly Withdrawal – Initial Monthly Withdrawal (This confirms the dollar amount of the increase).

---

## Phase II: Ongoing Monthly Management

The strategy provides a clear short-term roadmap for client expectations, expressed in dollar values, thus providing peace of mind during market volatility.

### Monthly Monitoring Protocol

The guardrails can be updated monthly, quarterly, or annually, but monthly monitoring provides the tightest control.

1.  **Determine Current Portfolio Value:** Obtain the current month-end portfolio value used for retirement funding.
2.  **Determine Current Withdrawal Amount:** Calculate the withdrawal amount for the coming month. This is the previous month's withdrawal amount adjusted for inflation.
3.  **Check Guardrail Status:** Compare the Current Portfolio Value against the pre-determined **Lower Guardrail Portfolio Value** and **Upper Guardrail Portfolio Value** established in Phase I.

### Withdrawal Calculation and Adjustment Scenarios (Edge Cases)

The withdrawal amount is the previous month's withdrawal adjusted for inflation, *unless* a guardrail is hit, which triggers a recalculation and a **permanent adjustment**.

#### Case 1: No Guardrail Hit (Portfolio Value is BETWEEN LGR and UGR)

**Action:** Continue current spending, adjusted only for inflation.

**Calculation:** The monthly withdrawal is increased by the assumed inflation rate since the last adjustment or recalculation.

*   *Note:* The sources emphasize that withdrawal rates used with historical analysis often assume inflation adjustments over time.

#### Case 2: Lower Guardrail Hit (Portfolio Value drops to or below LGR)

**Action:** Implement the spending reduction determined in Phase I.

**Calculation:** The new monthly withdrawal becomes the **Lower Guardrail Adjusted Monthly Withdrawal** amount (calculated in Step 3B).

**Impact Explanation (Client Communication):**

*   Communicate that the client has hit the predefined threshold (e.g., $901,000) that triggers a planned adjustment.
*   The spending cut (e.g., -$358 per month) is necessary to restore the plan’s safety margin (Chance of Underspending) back to the **Initial Target Success Rate** (e.g., 90%).
*   Emphasize that this adjustment is based on a plan established in advance using historical analysis.

#### Case 3: Upper Guardrail Hit (Portfolio Value grows to or above UGR)

**Action:** Implement the spending increase determined in Phase I.

**Calculation:** The new monthly withdrawal becomes the **Upper Guardrail Adjusted Monthly Withdrawal** amount (calculated in Step 4B).

**Impact Explanation (Client Communication):**

*   Communicate that the client has hit the predefined threshold (e.g., $1,188,000) that allows for a permanent spending increase.
*   The spending increase (e.g., +$567 per month) captures more "area under the curve" and restores the plan’s risk level back to the **Initial Target Success Rate** (e.g., 90%).

### Phase III: Recalculation and Subsequent Guardrails (Edge Case: Post-Adjustment)

A critical component of this dynamic strategy is that guardrails are a moving target, constantly changing due to time horizon, market movement, and other underlying assumptions.

**Action Required Upon Guardrail Hit:** **A full plan review and recalculation of new guardrails (Phase I) must be performed immediately after implementing a guardrail adjustment (Case 2 or Case 3)**.

**Rationale:** The existing guardrails are based on the risk profile of the *initial* spending amount. Once spending changes and the portfolio value shifts significantly, the risk profile has fundamentally altered, necessitating new thresholds for future adjustments.

**Recalculation Steps:**

1.  Use the **new portfolio value** (the value that triggered the guardrail) and the **new adjusted annual spending amount** (from Case 2 or Case 3) as the inputs for the subsequent Phase I calculations (Steps 2, 3, and 4).
2.  The resulting output will be a completely new set of dollar-based guardrail thresholds specific to the client’s updated financial reality and risk level.

### Summary of Communication (Advisor Focus)

Advisors should primarily communicate the guardrail system in terms of **dollar amounts** and **portfolio thresholds**, as this provides clearer guidance and peace of mind to clients than abstract concepts like probability percentages.

| Situation | Advisor Communication Language | Source Reference |
| :--- | :--- | :--- |
| **Initial Plan** | "We suggest starting at a spending level of X per month. We are targeting a Y% chance of success (or underspending)." | |
| **Market Drops** | "A cut in spending isn’t recommended until your portfolio falls all the way to **$LOWER GUARDRAIL AMOUNT** (e.g., $700,000)." | |
| **LGR Adjustment** | "If the portfolio reaches that level, the suggested spending decline is **$LOWER ADJUSTMENT AMOUNT** per month (e.g., -$500)." | |
| **Market Rises** | "If your portfolio grows to **$UPPER GUARDRAIL AMOUNT** (e.g., $1.2M), we can safely increase spending by **$UPPER ADJUSTMENT AMOUNT** per month (e.g., +$800)." | |
| **Standard Volatility** | "If the portfolio falls from $1M to $900,000, that decline is *not* one that merits a spending change, and you should continue your current spending." | |
