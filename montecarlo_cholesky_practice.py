# First of all we have to import the libraries that we are going to use

import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

yf.pdr_override()

# Then we have to download the data

tickers = ['AAPL', 'TSLA', 'MSFT', 'IBM']
startdate = '2012-01-01'
enddate = '2023-01-30'

data = pdr.get_data_yahoo(tickers, start=startdate, end=enddate)['Adj Close']

# Portfolio. weights = [0.x, 0.y, 0.z...]

num_stocks = len(tickers)

weights = np.random.random(num_stocks)
weights /= np.sum(weights)  # The sum = 1

#Variables

log_returns = np.log(1+data.pct_change()[1:])
mu = log_returns.mean()
var = log_returns.var()
drift = mu - (0.5*var)
covar = log_returns.cov()
stdev = log_returns.std()

# Inputs

trials = 100
days = 100

simulations = np.full(shape=(days, trials), fill_value=0.0)

porfolio = int(500000)

# Model

chol = np.linalg.cholesky(covar)
u = norm.ppf(np.random.rand(num_stocks, num_stocks))
Lu = chol.dot(u)

for i in range(0, trials):
    Z = norm.ppf(np.random.rand(days, num_stocks))
    daily_returns = np.inner(chol, drift.values + stdev.values * Z)
    simulations[:, i] = np.cumprod(np.inner(weights, daily_returns.T)+1)*porfolio
    simulations[0] = porfolio

# Graph 1

plt.figure(figsize=(15, 8))
plt.plot(simulations)
plt.ylabel('Portfolio Value')
plt.xlabel('Days')
plt.title('Montecarlo Simulation for a budget of ' + '$' + str(porfolio)  + '\n' + str(tickers) + '\n' + str(np.round(weights*100, 2)))

# Graph 2

sns.displot(pd.DataFrame(simulations).iloc[-1])
plt.ylabel('Frequency')
plt.xlabel('Portfolio')
plt.show()