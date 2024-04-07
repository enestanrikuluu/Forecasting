#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


#a
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Read the CSV file
df = pd.read_csv('renault.csv', header=None, names=['Date', 'Amount', 'Unwanted'])

# Drop the unwanted column
df.drop(columns=['Unwanted'], inplace=True)

# Convert the Date column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

print(df)

df.to_csv('renaultt.csv', encoding='utf-8', index=False)




# Plotting
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Amount'], marker='o', linestyle='-', color='b')
plt.title('Amount over Time')
plt.xlabel('Date')
plt.ylabel('Amount')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[2]:


# b naive forecasts

df['Ft_1'] = df['Amount'].shift(1)

df['Ft_12'] = df['Amount'].shift(12)

df_eval = df[(df['Date'] >= '2019-01-01') & (df['Date'] <= '2022-12-01')]


plt.figure(figsize=(14, 8))

# Original data
plt.plot(df_eval['Date'], df_eval['Amount'], label='Actual Data', color='blue', marker='o')

# Naive forecasts
plt.plot(df_eval['Date'][1:], df_eval['Ft_1'][1:], label='Ft = Dt−1 Forecast', linestyle='--', color='red')
plt.plot(df_eval['Date'][12:], df_eval['Ft_12'][12:], label='Ft = Dt−12 Forecast', linestyle='--', color='green')

plt.title('Actual vs. Naive Forecasts (Ft = Dt−1 and Ft = Dt−12)')
plt.xlabel('Date')
plt.ylabel('Amount')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


mae_1 = mean_absolute_error(df_eval['Amount'][1:], df_eval['Ft_1'][1:])
mape_1 = np.mean(np.abs((df_eval['Amount'][1:] - df_eval['Ft_1'][1:]) / df_eval['Amount'][1:])) * 100
rmse_1 = np.sqrt(mean_squared_error(df_eval['Amount'][1:], df_eval['Ft_1'][1:]))

mae_12 = mean_absolute_error(df_eval['Amount'][12:], df_eval['Ft_12'][12:])
mape_12 = np.mean(np.abs((df_eval['Amount'][12:] - df_eval['Ft_12'][12:]) / df_eval['Amount'][12:])) * 100
rmse_12 = np.sqrt(mean_squared_error(df_eval['Amount'][12:], df_eval['Ft_12'][12:]))


print("Error Metrics for Ft = Dt−1:")
print(f"MAE: {mae_1}")
print(f"MAPE: {mape_1}%")
print(f"RMSE: {rmse_1}\n")

print("Error Metrics for Ft = Dt−12:")
print(f"MAE: {mae_12}")
print(f"MAPE: {mape_12}%")
print(f"RMSE: {rmse_12}")


print(df)


# In[3]:


#c
df['3_MA_Forecast'] = df['Amount'].rolling(window=3).mean().shift(1)


df_adjusted_for_calculation = df[df['3_MA_Forecast'].notna()]

df_eval_2019_2022_adjusted = df_adjusted_for_calculation[(df_adjusted_for_calculation['Date'] >= '2019-01-01') & (df_adjusted_for_calculation['Date'] <= '2022-12-31')]

mae_3_MA_adjusted = mean_absolute_error(df_eval_2019_2022_adjusted['Amount'][3:], df_eval_2019_2022_adjusted['3_MA_Forecast'][3:])
mape_3_MA_adjusted = np.mean(np.abs((df_eval_2019_2022_adjusted['Amount'][3:] - df_eval_2019_2022_adjusted['3_MA_Forecast'][3:]) / df_eval_2019_2022_adjusted['Amount'][3:])) * 100
rmse_3_MA_adjusted = np.sqrt(mean_squared_error(df_eval_2019_2022_adjusted['Amount'][3:], df_eval_2019_2022_adjusted['3_MA_Forecast'][3:]))



df_eval_2014_2018 = df[(df['Date'] >= '2014-01-01') & (df['Date'] <= '2018-12-31')]


df_eval_2014_2018 = df_eval_2014_2018[df_eval_2014_2018['3_MA_Forecast'].notna()]

print(df_eval_2014_2018)


rmse_2014_2018 = np.sqrt(mean_squared_error(df_eval_2014_2018['Amount'], df_eval_2014_2018['3_MA_Forecast']))

print(rmse_2014_2018)

df_2022 = df_eval_2019_2022_adjusted[(df_eval_2019_2022_adjusted['Date'] >= '2022-01-01') & (df_eval_2019_2022_adjusted['Date'] <= '2022-12-31')].copy()
z_score = 1.645  
df_2022['Lower_PI'] = df_2022['3_MA_Forecast'] - (z_score * rmse_2014_2018)
df_2022['Upper_PI'] = df_2022['3_MA_Forecast'] + (z_score * rmse_2014_2018)



# Plotting
plt.figure(figsize=(14, 8))
plt.plot(df_eval_2019_2022_adjusted['Date'], df_eval_2019_2022_adjusted['Amount'], label='Actual Data', color='blue', marker='o')
plt.plot(df_eval_2019_2022_adjusted['Date'][3:], df_eval_2019_2022_adjusted['3_MA_Forecast'][3:], label='3-Period MA Forecast', linestyle='--', color='red')

plt.fill_between(df_2022['Date'], df_2022['Lower_PI'], df_2022['Upper_PI'], color='gray', alpha=0.2, label='90% Prediction Interval')

plt.title('Actual Data vs. 3-Period Moving Average Forecast (Adjusted for 2019-2022)')
plt.xlabel('Date')
plt.ylabel('Amount')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print('MAE:', mae_3_MA_adjusted)
print('MAPE:', mape_3_MA_adjusted)
print('RMSE:', rmse_3_MA_adjusted)


# In[4]:


#e
from statsmodels.tsa.api import SimpleExpSmoothing
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


df_evaluated = pd.read_csv('renault.csv', header=None, names=['Date', 'Amount', 'Unwanted'])

df_evaluated.drop(columns=['Unwanted', 'Date'], inplace=True)

results = {}


alphas = np.arange(0.1, 1.1, 0.1)
for alpha in alphas:
    
    model = SimpleExpSmoothing(df_evaluated).fit(smoothing_level=alpha, optimized=False)
    df['Simple_Exp_Forecast'] = model.fittedvalues

    df_eval = df[(df['Date'] >= '2019-01-01') & (df['Date'] <= '2022-12-01')]

    mae = mean_absolute_error(df_eval['Amount'], df_eval['Simple_Exp_Forecast'])
    mape = np.mean(np.abs((df_eval['Amount'] - df_eval['Simple_Exp_Forecast']) / df_eval['Amount'])) * 100
    rmse = np.sqrt(mean_squared_error(df_eval['Amount'], df_eval['Simple_Exp_Forecast']))
    results[alpha] = (mae, mape, rmse)
    

    


print(results)

best_alpha = min(results, key=lambda x: results[x][2])

model = SimpleExpSmoothing(df_evaluated).fit(smoothing_level=best_alpha, optimized=False)
df['Simple_Exp_Forecast'] = model.fittedvalues

df_eval = df[(df['Date'] >= '2019-01-01') & (df['Date'] <= '2022-12-01')]

mae = mean_absolute_error(df_eval['Amount'], df_eval['Simple_Exp_Forecast'])
mape = np.mean(np.abs((df_eval['Amount'] - df_eval['Simple_Exp_Forecast']) / df_eval['Amount'])) * 100
rmse = np.sqrt(mean_squared_error(df_eval['Amount'], df_eval['Simple_Exp_Forecast']))





print(f'Best Alpha: {best_alpha}')
print(f'MAE: {results[best_alpha][0]}')
print(f'MAPE: {results[best_alpha][1]}')
print(f'RMSE: {results[best_alpha][2]}')


df_eval_2014_2018 = df[(df['Date'] >= '2014-01-01') & (df['Date'] <= '2018-12-31')]
df_eval_2014_2018_rmse = np.sqrt(mean_squared_error(df_eval_2014_2018['Amount'], df_eval_2014_2018['Simple_Exp_Forecast']))
    
z_score = 1.645  # For 90% confidence interval
df_2022 = df[(df['Date'] >= '2022-01-01') & (df['Date'] <= '2022-12-31')]
df_2022['Lower_PI'] = df_2022['Simple_Exp_Forecast'] - z_score * df_eval_2014_2018_rmse
df_2022['Upper_PI'] = df_2022['Simple_Exp_Forecast'] + z_score * df_eval_2014_2018_rmse



plt.figure(figsize=(14, 8))

plt.plot(df_eval['Date'], df_eval['Amount'], label='Actual Data', color='blue', marker='o', markersize=5)


plt.plot(df_2022['Date'], df_2022['Simple_Exp_Forecast'], label='Exponential Smoothing Forecast', color='red', linestyle='--')


plt.fill_between(df_2022['Date'], df_2022['Lower_PI'], df_2022['Upper_PI'], color='gray', alpha=0.2, label='90% Prediction Interval')

plt.title('Actual Data and Exponential Smoothing Forecast with Prediction Intervals (2022)')
plt.xlabel('Date')
plt.ylabel('Amount')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# In[5]:


#f
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error


df_evaluated = pd.read_csv('renault.csv', header=None, names=['Date', 'Amount', 'Unwanted'])

df_evaluated.drop(columns=['Unwanted', 'Date'], inplace=True)

results = {}

for alpha in np.arange(0.1, 1.1, 0.1):
    for beta in np.arange(0.1, 1.1, 0.1):
        
        model = ExponentialSmoothing(df_evaluated, trend='add', seasonal=None, initialization_method="estimated").fit(smoothing_level=alpha, smoothing_trend=beta)
        
        df['Double_Exp_Forecast'] = model.fittedvalues

        df_eval = df[(df['Date'] >= '2019-01-01') & (df['Date'] <= '2022-12-01')]

        mae = mean_absolute_error(df_eval['Amount'], df_eval['Double_Exp_Forecast'])
        mape = np.mean(np.abs((df_eval['Amount'] - df_eval['Double_Exp_Forecast']) / df_eval['Amount'])) * 100
        rmse = np.sqrt(mean_squared_error(df_eval['Amount'], df_eval['Double_Exp_Forecast']))
        results[(alpha, beta)] = (mae, mape, rmse)



    



best_alpha_beta = min(results, key=lambda x: results[x][2])

alpha, beta = best_alpha_beta


model = ExponentialSmoothing(df_evaluated, trend='add', seasonal=None, initialization_method="estimated").fit(smoothing_level=alpha, smoothing_trend=beta)

df['Double_Exp_Forecast'] = model.fittedvalues

df_eval = df[(df['Date'] >= '2019-01-01') & (df['Date'] <= '2022-12-01')]

mae = mean_absolute_error(df_eval['Amount'], df_eval['Double_Exp_Forecast'])
mape = np.mean(np.abs((df_eval['Amount'] - df_eval['Double_Exp_Forecast']) / df_eval['Amount'])) * 100
rmse = np.sqrt(mean_squared_error(df_eval['Amount'], df_eval['Double_Exp_Forecast']))




print(f'Best Alpha, Beta: {best_alpha_beta}')
print(f'MAE: {results[best_alpha_beta][0]}')
print(f'MAPE: {results[best_alpha_beta][1]}')
print(f'RMSE: {results[best_alpha_beta][2]}')


df_eval_2014_2018 = df[(df['Date'] >= '2014-01-01') & (df['Date'] <= '2018-12-31')]
df_eval_2014_2018_rmse = np.sqrt(mean_squared_error(df_eval_2014_2018['Amount'], df_eval_2014_2018['Double_Exp_Forecast']))
    

z_score = 1.645  # For 90% confidence interval
df_2022 = df[(df['Date'] >= '2022-01-01') & (df['Date'] <= '2022-12-31')]
df_2022['Lower_PI'] = df_2022['Double_Exp_Forecast'] - z_score * df_eval_2014_2018_rmse
df_2022['Upper_PI'] = df_2022['Double_Exp_Forecast'] + z_score * df_eval_2014_2018_rmse


plt.figure(figsize=(14, 8))

plt.plot(df_eval['Date'], df_eval['Amount'], label='Actual Data', color='blue', marker='o', markersize=5)



plt.plot(df_2022['Date'], df_2022['Double_Exp_Forecast'], label='Exponential Smoothing Forecast', color='red', linestyle='--')

plt.fill_between(df_2022['Date'], df_2022['Lower_PI'], df_2022['Upper_PI'], color='gray', alpha=0.2, label='90% Prediction Interval')

plt.title('Actual Data and Exponential Smoothing Forecast with Prediction Intervals (2022)')
plt.xlabel('Date')
plt.ylabel('Amount')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# In[6]:


#h
from statsmodels.tsa.holtwinters import ExponentialSmoothing





df_forecast = pd.DataFrame()
df_forecast['12-Month-Forecast'] = model.forecast(12)

df_forecast['Forecast_3m_ahead'] = df_forecast['12-Month-Forecast'].shift(-3)
df_forecast['Forecast_6m_ahead'] = df_forecast['12-Month-Forecast'].shift(-6)




df_2022 = df[(df['Date'] >= '2022-01-01') & (df['Date'] <= '2022-12-31')]

print(df_2022['Amount'][:9].to_numpy())
print(df_forecast['Forecast_3m_ahead'][:9].to_numpy())
print()


mae_3m = mean_absolute_error(df_2022['Amount'][:9], df_forecast['Forecast_3m_ahead'][:9])
mape_3m = np.mean(np.abs((np.subtract(df_forecast['Forecast_3m_ahead'][:9].to_numpy(), df_2022['Amount'][:9].to_numpy())) / df_2022['Amount'][:9].to_numpy())) * 100
rmse_3m = np.sqrt(mean_squared_error(df_2022['Amount'][:9], df_forecast['Forecast_3m_ahead'][:9]))

mae_6m = mean_absolute_error(df_2022['Amount'][:6], df_forecast['Forecast_6m_ahead'][:6])
mape_6m = np.mean(np.abs((np.subtract(df_forecast['Forecast_6m_ahead'][:6].to_numpy(), df_2022['Amount'][:6].to_numpy())) / df_2022['Amount'][:6].to_numpy())) * 100
rmse_6m = np.sqrt(mean_squared_error(df_2022['Amount'][:6], df_forecast['Forecast_6m_ahead'][:6]))


print("3-Month Ahead Forecast Errors:")
print(f"MAE: {mae_3m}, MAPE: {mape_3m}%, RMSE: {rmse_3m}\n")

print("6-Month Ahead Forecast Errors:")
print(f"MAE: {mae_6m}, MAPE: {mape_6m}%, RMSE: {rmse_6m}")


# In[ ]:





# In[7]:


#i
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error


df = pd.read_csv('renault.csv', header=None, names=['Date', 'Amount', 'Unwanted'])
df.drop(columns=['Unwanted'], inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)


df['U_t'] = df['Amount'] - df['Amount'].shift(12)


plt.figure(figsize=(12, 6))
plt.plot(df.index, df['U_t'], label='Transformed Series (U_t)')
plt.title('Detrended Time Series (U_t)')
plt.xlabel('Date')
plt.ylabel('U_t')
plt.legend()
plt.show()




alphas = np.linspace(0.1, 1.0, 10)
best_alpha = None
lowest_mae = float('inf')
forecast_period = 12  


performance = pd.DataFrame(columns=['Alpha', 'MAE', 'MAPE', 'RMSE'])

for alpha in alphas:

    model = SimpleExpSmoothing(df['U_t'].dropna(), initialization_method="estimated").fit(smoothing_level=alpha)

    df['G_t'] = model.fittedvalues

  
    df['F_t'] = df['G_t'].shift(-12) + df['Amount'].shift(12)  

    df_eval = df.loc['2019-01-01':'2022-12-31'].dropna(subset=['F_t', 'Amount'])
    mae = mean_absolute_error(df_eval['Amount'], df_eval['F_t'])
    mape = np.mean(np.abs((df_eval['Amount'] - df_eval['F_t']) / df_eval['Amount'])) * 100
    rmse = np.sqrt(mean_squared_error(df_eval['Amount'], df_eval['F_t']))
    performance = performance.append({'Alpha': alpha, 'MAE': mae, 'MAPE': mape, 'RMSE': rmse}, ignore_index=True)
    
    if mae < lowest_mae:
        best_alpha = alpha
        lowest_mae = mae

print("Best Alpha:", best_alpha)


print(performance[performance['Alpha'] == best_alpha])


# In[ ]:





# In[24]:


#2.a

df = pd.read_csv('domestic_beer_sales.csv', header=None, names=['Month', 'Sales'], index_col=False)
df
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Sales'], marker='o', linestyle='-', color='b')
plt.title('Monthly Sales of Beer in Turkey (2010-2014)')
plt.xlabel('Months')
plt.ylabel('Sales')
plt.grid(True)
plt.xticks(rotation=45)  
plt.tight_layout()
plt.show()


# In[25]:


#b

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error



df['Ft_1'] = df['Sales'].shift(1)

df['Ft_12'] = df['Sales'].shift(12)


df_eval = df[(df['Month'] >= 13) & (df['Month'] <= 60)]


mae_1 = mean_absolute_error(df_eval['Sales'][1:], df_eval['Ft_1'][1:])
mape_1 = np.mean(np.abs((df_eval['Sales'][1:] - df_eval['Ft_1'][1:]) / df_eval['Sales'][1:])) * 100
rmse_1 = np.sqrt(mean_squared_error(df_eval['Sales'][1:], df_eval['Ft_1'][1:]))

mae_12 = mean_absolute_error(df_eval['Sales'][12:], df_eval['Ft_12'][12:])
mape_12 = np.mean(np.abs((df_eval['Sales'][12:] - df_eval['Ft_12'][12:]) / df_eval['Sales'][12:])) * 100
rmse_12 = np.sqrt(mean_squared_error(df_eval['Sales'][12:], df_eval['Ft_12'][12:]))

print("Naive Forecast Ft=Dt−1 Error Metrics:")
print("MAE:", mae_1)
print("MAPE:", mape_1)
print("RMSE:", rmse_1)

print("\nNaive Forecast Ft=Dt−12 Error Metrics:")
print("MAE:", mae_12)
print("MAPE:", mape_12)
print("RMSE:", rmse_12)

print(df)


# In[27]:


#d
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error


df['Month'] = pd.date_range(start='2010-01-01', periods=len(df), freq='M')

df.set_index('Month', inplace=True)
df = df.asfreq('M')


alphas = np.linspace(0.1, 1, 10)
betas = np.linspace(0.1, 1, 10)
gammas = [0.1, 0.5, 0.9]
seasonal_periods = 12  

best_params = {}
lowest_error = float('inf')


for alpha in alphas:
    for beta in betas:
        for gamma in gammas:
            model = ExponentialSmoothing(df['Sales'], trend='add', seasonal='add', seasonal_periods=seasonal_periods,
                                         initialization_method="estimated").fit(smoothing_level=alpha, smoothing_slope=beta, smoothing_seasonal=gamma)
            df['Forecast'] = model.fittedvalues
            
            df_eval = df['2011-01-01':'2014-12-31']
            mae = mean_absolute_error(df_eval['Sales'], df_eval['Forecast'])
            mape = np.mean(np.abs((df_eval['Sales'] - df_eval['Forecast']) / df_eval['Sales'])) * 100
            rmse = np.sqrt(mean_squared_error(df_eval['Sales'], df_eval['Forecast']))
            
            if rmse < lowest_error:
                lowest_error = rmse
                best_params = {'alpha': alpha, 'beta': beta, 'gamma': gamma, 'MAE': mae, 'MAPE': mape, 'RMSE': rmse}


print("Best parameters:", best_params)

df_2011_2013 = df['2011-01-01':'2013-12-31']
rmse_2011_2013 = np.sqrt(mean_squared_error(df_2011_2013['Sales'], df_2011_2013['Forecast']))


z_score = 1.645  # for 90% confidence
df_2014 = df['2014-01-01':'2014-12-31']
df_2014['Lower_PI'] = df_2014['Forecast'] - z_score * rmse_2011_2013
df_2014['Upper_PI'] = df_2014['Forecast'] + z_score * rmse_2011_2013

print("Prediction intervals for 2014:")
print(df_2014[['Forecast', 'Lower_PI', 'Upper_PI']])


import matplotlib.pyplot as plt
import matplotlib.dates as mdates


plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Sales'], label='Actual Sales', color='blue')
plt.plot(df_2014.index, df_2014['Forecast'], label='Forecast', color='red', linestyle='--')

plt.fill_between(df_2014.index, df_2014['Lower_PI'], df_2014['Upper_PI'], color='grey', alpha=0.3, label='90% Prediction Interval')


plt.axvline(x=pd.to_datetime('2014-01-01'), color='green', linestyle='--', linewidth=0.8, label='Start of 2014')

plt.title('Beer Sales Forecast with Prediction Intervals for 2014')
plt.xlabel('Year')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)


plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.gcf().autofmt_xdate()  

plt.show()


# In[59]:


#e
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Assuming df is your DataFrame loaded with actual sales data up to the end of 2013

model = ExponentialSmoothing(df['Sales'], trend='add', seasonal='add', seasonal_periods=12, initialization_method="estimated").fit(smoothing_level=0.1, smoothing_slope=0.7, smoothing_seasonal=0.9)


forecasts = model.forecast(18)


df_2014 = df['2014-01-01':'2014-12-31']


forecast_3m_ahead = forecasts[2:14]  
forecast_6m_ahead = forecasts[5:14]  




mae_3m = mean_absolute_error(df_2014['Sales'][2:], forecast_3m_ahead[:-2])
mape_3m = np.mean(np.abs((np.subtract(df_2014['Sales'][2:].to_numpy(), forecast_3m_ahead[:-2].to_numpy())) / df_2014['Sales'][2:].to_numpy())) * 100
rmse_3m = np.sqrt(mean_squared_error(df_2014['Sales'][2:], forecast_3m_ahead[:-2]))






mae_6m = mean_absolute_error(df_2014['Sales'][5:], forecast_6m_ahead[:-2])
mape_6m = np.mean(np.abs((np.subtract(df_2014['Sales'][5:].to_numpy(), forecast_6m_ahead[:-2].to_numpy())) / df_2014['Sales'][5:].to_numpy())) * 100
rmse_6m = np.sqrt(mean_squared_error(df_2014['Sales'][5:], forecast_6m_ahead[:-2]))

print("3-Month Ahead Forecast Error Metrics:")
print(f"MAE: {mae_3m}, MAPE: {mape_3m}%, RMSE: {rmse_3m}")

print("\n6-Month Ahead Forecast Error Metrics:")
print(f"MAE: {mae_6m}, MAPE: {mape_6m}%, RMSE: {rmse_6m}")


# In[ ]:





# In[ ]:




