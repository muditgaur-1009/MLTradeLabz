
# Gym stuff
import gym
import gym_anytrading
from gym_anytrading.envs import StocksEnv

# Stable baselines - rl stuff
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C

#Quant Finance
from finta import TA
import quantstats as qs

# Processing libraries
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



import pandas as pd

# key = "your_api_key"  # Replace with your AlphaVantage API key
key=""

# Define the URLs for each month
month_1 = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=TSLA&interval=1min&slice=year1month1&apikey={key}&datatype=csv'
month_2 = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=TSLA&interval=1min&slice=year1month2&apikey={key}&datatype=csv'
month_3 = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=TSLA&interval=1min&slice=year1month3&apikey={key}&datatype=csv'
month_4=f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=TSLA&interval=1min&slice=year1month4&apikey={key}&datatype=csv'
# Read the CSV data for each month

csv_1 = pd.read_csv(month_1)
csv_2 = pd.read_csv(month_2)
csv_3 = pd.read_csv(month_3)
csv_4=pd.read_csv(month_4)

# Concatenate the DataFrames into one
data = pd.concat([csv_1, csv_2, csv_3,csv_4])

# Save the DataFrame as a CSV file
data.to_csv('data.csv', index=False)

data.head(3)

data.info()

data = data.rename(columns = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})

data['timestamp'] = pd.to_datetime(data['timestamp'])
data.set_index('timestamp', inplace=True)
data.sort_values('timestamp', ascending=True, inplace=True)

data.head()

"""## Add Custom Indicators"""

data['return'] = np.log(data['Close'] / data['Close'].shift(1))

#Create columns for technical indicators & add them to the dataframe
data['RSI'] = TA.RSI(data,16)
data['SMA'] = TA.SMA(data, 20)
data['SMA_L'] = TA.SMA(data, 41)
# data['OBV'] = TA.OBV(data)
data['VWAP'] = TA.VWAP(data)
data['EMA'] = TA.EMA(data)
data['ATR'] = TA.ATR(data)
data.fillna(0, inplace=True)

#Add momentum, volatitlity, & distance to the data_frame
data['momentum'] = data['return'].rolling(5).mean().shift(1)
data['volatility'] = data['return'].rolling(20).std().shift(1)
data['distance'] = (data['Close'] - data['Close'].rolling(50).mean()).shift(1)

#Perform a simple linear regression direction prediction
lags = 5

cols = []
for lag in range(1, lags + 1):
  col = f'lag_{lag}'
  data[col] = data['Close'].shift(lag)
  cols.append(col)

data.dropna(inplace=True)

reg = np.linalg.lstsq(data[cols], data['Close'], rcond=None)[0]
data['Prediction'] = np.dot(data[cols], reg)

data.tail()

#Create a function to properly format data frame to be passed through environment
def signals(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:,'Close'].to_numpy()[start:end]
    signal_features = env.df.loc[:, ['Open','High','Low','Close','Volume','return','momentum','volatility','distance','RSI','SMA','SMA_L','VWAP','EMA','ATR', 'Prediction']].to_numpy()[start:end]
    return prices, signal_features

#Replace default data process with custom function from above
class MyCustomEnv(StocksEnv):
    _process_data = signals

#Initialize an environment setting the window size and train data
window_size = 65
start_index = window_size
end_train_index = round(len(data)*0.70)
end_val_index = len(data)

env2 = MyCustomEnv(df=data, window_size=window_size, frame_bound=(start_index, end_train_index))

#Create a Dummy Vector of our environment
env_maker = lambda: env2
env = DummyVecEnv([env_maker])

"""## Train Test"""

#initialize our model and train
policy_kwargs = dict(optimizer_class='RMSpropTFLike', optimizer_kwargs=dict(eps=1e-5))
actor_critic = A2C('MlpPolicy', env, verbose=1)
actor_critic.learn(total_timesteps=500000)

#Create a new environment with validation data
env = MyCustomEnv(df=data, window_size=window_size, frame_bound=(end_train_index, end_val_index))
obs = env.reset()

while True:
    obs = obs[np.newaxis, ...]
    action, _states = actor_critic.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        print("info", info)
        break

#Plot the results
plt.figure(figsize=(16,9))
env.render_all()
plt.show()



net_worth = []
obs = env.reset()
while True:
    obs = obs[np.newaxis, ...]
    action, _ = actor_critic.predict(obs)  # Removed _states variable
    obs, rewards, done, info = env.step(action)
    net_worth.append(env.history['total_profit'])
    if done:
        break

net_worth = np.array(net_worth)
returns = pd.Series(np.diff(net_worth) / net_worth[:-1].T.flatten())

returns = np.diff(net_worth) / net_worth[:-1].flatten()

# Convert returns to a Pandas Series
returns = pd.Series(returns)

# Calculate CAGR
if len(returns) < 2:
    print("Insufficient data points to calculate CAGR.")
else:
    cagr = qs.stats.cagr(returns)
    print("CAGR:", cagr)
