#Portfolio optimisation (2 year timescale) compared to SP500

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats

# Loads data
pickle_filename = "stock_analysis_data.pkl"
file = open(pickle_filename, 'rb')
loaded_data = pickle.load(file)
file.close()
df = pd.DataFrame.from_dict(loaded_data["stock_data"])
df.index = pd.to_datetime(loaded_data["times"])
pickle_filename = "SPY_analysis_data.pkl"
file = open(pickle_filename, 'rb')
loaded_data = pickle.load(file)
file.close()
df_SPY = pd.DataFrame.from_dict(loaded_data["SPY_data"])
df_SPY.index = pd.to_datetime(loaded_data["times"])

# Log returns dfs
df_log_returns = (np.log(df).diff()).dropna() # type: ignore
df_SPY_log_returns = (np.log(df_SPY).diff()).dropna().sum(axis=1) # type: ignore

# Calculate mean return and covaraince matrix, plots histograms of returns
mean_returns = df_log_returns.mean()
cov_returns = df_log_returns.cov() 
df_log_returns.hist(figsize = (10 ,5), bins = 100)
plt.tight_layout()

# Monte Carlo simulation for portfolio performance measured by Sharpe Ratio
risk_free_rate = 0.0446 # 17/06/25 10Y US Treasury yield
annual_log_rfr = np.log(1 + risk_free_rate)
daily_rf_rate = ((1 + risk_free_rate) ** (1/365)) - 1
market_vol = df_SPY_log_returns.std() * np.sqrt(252)
simulations = 100000
all_weights = np.zeros((simulations, len(list(df))))
portfolio_return = np.zeros(simulations)
portfolio_vol = np.zeros(simulations)
portfolio_sharpe = np.zeros(simulations)
portfolio_beta = np.zeros(simulations)
portfolio_alpha = np.zeros(simulations)
portfolio_treynor = np.zeros(simulations)
portfolio_m2 = np.zeros(simulations)
portfolio_ = np.zeros(simulations)

# Optimisation and performance metrics calculation
for i in range(simulations):
    weights = np.array(np.random.random(len(list(df))))
    weights = weights/np.sum(weights)  #type:ignore
    all_weights[i,:] = weights
    portfolio_return[i] = np.sum(mean_returns * weights) * 252
    portfolio_vol[i] = np.sqrt(np.transpose(weights) @ cov_returns @ weights) * np.sqrt(252)
    portfolio_sharpe[i] = (portfolio_return[i] - annual_log_rfr) / portfolio_vol[i]
    portfolio_log_return = (df_log_returns * weights).sum(axis = 1) # type: ignore
    # Calculates alpha and beta compared to SPY over period
    (beta, alpha) = stats.linregress(df_SPY_log_returns.values,
                    portfolio_log_return.values)[0:2]
    portfolio_beta[i] = beta # type: ignore
    portfolio_alpha[i] = alpha # type: ignore
    portfolio_treynor[i] = (portfolio_return[i] - annual_log_rfr) / beta
    portfolio_m2[i] = portfolio_sharpe[i] * market_vol + annual_log_rfr
    print(f'Simulation {i} completed.')

# Creating dataframe showing candidate optimal portfolios according to metrics
measures = {'Sharpe': portfolio_sharpe, 'Alpha': portfolio_alpha, 'Treynor': portfolio_treynor, 'M2': portfolio_m2}
measure_names = list(measures.keys())
optimal_portfolios_index = ['Simulation ID'] + measure_names + ['Return', 'Volatility', 'Beta']
optimal_portfolios_data = {}

# Iterate through each measure to construct dataframe
for measure in measure_names:
    current_measure = measures[measure]
    
    max_value = current_measure.max()
    max_index = current_measure.argmax()

    column_values = []

    # Populate based on optimal_portfolios_index
    for index_label in optimal_portfolios_index:
        if index_label == 'Simulation ID':
            column_values.append(max_index)
        elif index_label in measure_names:
            column_values.append(measures[index_label][max_index])
        elif index_label == 'Return':
            column_values.append(portfolio_return[max_index])
        elif index_label == 'Volatility':
            column_values.append(portfolio_vol[max_index])
        elif index_label == 'Beta':
            column_values.append(portfolio_beta[max_index])

    # Add the constructed column to dataframe dictionary
    optimal_portfolios_data[measure] = column_values

# Dataframe of candidate optimal portfolios
optimal_portfolios = pd.DataFrame(optimal_portfolios_data, index=optimal_portfolios_index)
print(optimal_portfolios)

# Plots simulated portfolios indicating highest Sharpe portfolio and Capital Allocation Line
max_vol = portfolio_vol.max()
max_sharpe_id = measures['Sharpe'].argmax()
plt.figure(figsize=(15,10))
plt.scatter(portfolio_vol, portfolio_return, c=portfolio_sharpe, cmap='plasma')
x = np.linspace(0, max_vol + 0.05)
y = portfolio_sharpe.max() * x + annual_log_rfr
plt.plot(x,y)
plt.colorbar(label = 'Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Log Portfolio Return')
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.scatter(portfolio_vol[max_sharpe_id], portfolio_return[max_sharpe_id], c = 'red', s = 20, edgecolors='black')
plt.show()
