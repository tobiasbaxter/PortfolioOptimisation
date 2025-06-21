# Portfolio Optimisation & Performance Analysis

This Python script performs a Monte Carlo simulation to identify optimal investment portfolios based on various performance metrics, including Sharpe Ratio, Alpha, Treynor Ratio, and M2 (Modigliani-Modigliani Measure). It evaluates portfolio performance over a 2-year timescale and compares it against the S&P 500 (SPY).

## Methodology Overview

The script follows a structured approach to portfolio analysis and optimisation:

* **Data Loading & Preprocessing:**
    * Loads historical stock data from `stock_analysis_data.pkl` and S&P 500 (SPY) data from `SPY_analysis_data.pkl` using Python's `pickle` module.
    * Converts the loaded dictionaries into Pandas DataFrames and sets their indices to datetime objects.
    * Calculates daily logarithmic returns for both individual stocks and the SPY index.

* **Initial Statistical Analysis:**
    * Computes the mean daily logarithmic returns for each stock and the covariance matrix of these returns.
    * Generates and displays histograms of the individual stock logarithmic returns to visualize their distributions.

* **Monte Carlo Simulation Setup:**
    * Defines key financial parameters, including a risk-free rate (0.0446, based on 10Y US Treasury yield as of 17/06/25) and the market volatility derived from SPY returns.
    * Sets up a large number of simulations (100,000) to explore a wide range of possible portfolio weight combinations.
    * Initialises NumPy arrays to store simulation results for portfolio returns, volatility, Sharpe Ratio, Beta, Alpha, Treynor Ratio, and M2.

* **Portfolio Optimisation Loop:**
    * For each simulation, random weights are generated for the assets and then normalised to sum to 1.
    * The script calculates the annualised portfolio return and volatility based on these weights.
    * Key performance metrics are computed:
        * **Sharpe Ratio:** Measures risk-adjusted return relative to the risk-free rate.
        * **Alpha:** Measures the excess return of a portfolio relative to its expected return, given its beta and the market's return. Calculated using linear regression against SPY returns.
        * **Treynor Ratio:** Measures return earned in excess of that which could have been earned on a riskless asset per unit of market risk.
        * **M2 (Modigliani-Modigliani Measure):** Adjusts the portfolio's volatility to match that of the market and then compares its return.
    * The progress of the simulations is printed to the console.

* **Optimal Portfolio Dataframe Creation:**
    * Identifies the simulation ID (portfolio) that maximised each of the four key performance measures (Sharpe, Alpha, Treynor, M2) independently.
    * Constructs a Pandas DataFrame (`optimal_portfolios`) that presents the details of these candidate optimal portfolios. Each column corresponds to a measure (e.g., 'Sharpe' column shows the portfolio that maximised Sharpe Ratio).
    * The DataFrame includes:
        * The **Simulation ID** for the optimal portfolio.
        * The **maximum value** of the specific measure for that column.
        * The values of **all other performance measures** (Sharpe, Alpha, Treynor, M2) at that same optimal portfolio's simulation ID.
        * The **Return, Volatility, and Beta** of that specific optimal portfolio.

* **Visualisation:**
    * Generates a scatter plot of all simulated portfolios, showing their annualised volatility versus annualised return.
    * The points on the scatter plot are coloured by their respective Sharpe Ratios, providing a visual representation of risk-adjusted performance.
    * Draws the Capital Allocation Line (CAL), which illustrates the trade-off between risk and return for efficient portfolios when combined with a risk-free asset.
    * Highlights the portfolio with the highest Sharpe Ratio on the plot for easy identification.

## How to Use the Code

1.  **Ensure Data Files:** Place `stock_analysis_data.pkl` and `SPY_analysis_data.pkl` in the same directory as the Python script. These files are crucial for the script to run.

2.  **Run the Script:** Execute the Python script using a Python interpreter (e.g., `python your_script_name.py`).

3.  **Review Output:**
    * Histograms of log returns will be displayed (close the plot to continue execution).
    * The `optimal_portfolios` DataFrame will be printed to your console, showing the characteristics of the portfolios that optimised each chosen metric.
    * A scatter plot visualising all simulated portfolios and the Capital Allocation Line will be displayed (close the plot to finish execution).

## Technical Stack

* **Data Manipulation:** Pandas, NumPy
* **Statistical Analysis:** SciPy (specifically `scipy.stats` for linear regression), NumPy
* **Plotting:** Matplotlib
* **Data Serialization:** Pickle

## Limitations and Further Analyses

* **Fixed Data Source:** The script relies on pre-saved `.pkl` files. For real-time analysis, data retrieval would need to be integrated, potentially using a financial API.

* **Monte Carlo Randomness:** The Monte Carlo simulation relies on random weight generation. While 100,000 simulations provide a good sample, deterministic optimisation methods (e.g., quadratic programming for mean-variance optimisation) could be used for more precise efficient frontier calculations.

* **Simple Risk-Free Rate:** A static risk-free rate is used. In a more sophisticated model, this could be dynamically updated or derived from yield curves.

* **Performance Metric Scope:** The script focuses on standard metrics. Other advanced metrics (e.g., Sortino Ratio, Conditional Value at Risk) could be incorporated.

* **No Constraints on Weights:** The current weight generation does not include constraints (e.g., no short-selling, minimum/maximum allocation per asset). Adding such constraints would make the optimisation more realistic.

* **Single Period Optimisation:** This is a single-period optimisation. Multi-period or dynamic rebalancing strategies are not considered.

* **Historical Data Assumption:** The analysis assumes that historical returns and volatilities are representative of future performance, which is a common simplification in portfolio theory but has inherent limitations in practice.