import streamlit

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from textblob import TextBlob

# Key Sentiment indicators from reports
sample_text = """
1. Infrastructure Bond (IFB) oversubscribed - CBK raised Ksh240.96 Billion vs target of Ksh70 billion.
2. Foreign investors invested in IFB, attracted by 18.4% yield, increasing USD supply. Demand for KES rose.
3. New $1.5 Bn Eurobond boosted foreign investor confidence in Kenyan economy.
4. USD/KES volatility due to increased USD supply & low demand. Foreign investors sought KES for IFB.
5. USD hoarders and investors seeking KES increased KES demand, dropping USD demand.
6. KES appreciation against USD excited markets. USD hoarders panicked, selling dollars.
7. Exchange rate will settle at levels of KES 150s when the dust settles!
8. Global money laundering agency officially puts Kenya on grey list
"""

# Load data
try:
    df = pd.read_excel("TRADE WEIGHTED AVERAGE INDICATIVE CURRENT RATES (3).xlsx")
except FileNotFoundError:
    print("Error: File not found.")
    exit()
except Exception as e:
    print("Error:", e)
    exit()

# Convert Date column to datetime with format dd/mm/yyyy
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

# Set Date column as index
df.set_index('Date', inplace=True)

# Sort by Date
df.sort_index(inplace=True)

# Extracting only necessary columns
df = df[['Mean']]

# Check for missing data
if df.isnull().values.any():
    print("Error: Missing data found.")
    exit()

# Incorporate inflation rate, bond rate, and CBK rate into the model
inflation_rate = 1.075  # 7.5% inflation rate
bond_rate = 1.166784  # 16.6784% bond rate
cbk_rate = 1.13  # 13% CBK rate

# Adjust the data using the inflation rate, bond rate, and CBK rate
df_adjusted = df

# Fit SARIMAX model
try:
    model = SARIMAX(df_adjusted, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit()
except Exception as e:
    print("Error fitting SARIMAX model:", e)
    exit()

# Forecast exchange rates for the next 7 days
try:
    forecast_7days = model_fit.forecast(steps=7)
except Exception as e:
    print("Error forecasting for the next 7 days:", e)
    exit()

# Adjust forecasted rates for Friday to carry across Saturday and Sunday
# Generate forecast dates for the next 7 days including Monday
forecast_dates_7days = pd.date_range(start=df_adjusted.index[-1] + pd.Timedelta(days=1), periods=7)

# Print forecasted exchange rates for the next 7 days
forecast_df_7days = pd.DataFrame({'Date': forecast_dates_7days, 'Forecasted Exchange Rate': forecast_7days[:7]})

# Adjust forecasted rates for Friday to carry across Saturday and Sunday
friday_rate = forecast_df_7days.loc[forecast_df_7days['Date'].dt.dayofweek == 4, 'Forecasted Exchange Rate'].values[0]
forecast_df_7days.loc[forecast_df_7days['Date'].dt.dayofweek.isin([5, 6]), 'Forecasted Exchange Rate'] = friday_rate

print("Forecasted Exchange Rates for the next 7 days:")
print(forecast_df_7days.to_string(index=False))

# Perform sentiment analysis on the sample text
sentiment = TextBlob(sample_text).sentiment.polarity

# Determine sentiment label
if sentiment > 0:
    sentiment_label = 'Positive'
elif sentiment < 0:
    sentiment_label = 'Negative'
else:
    sentiment_label = 'Neutral'

# Print overall sentiment
print("\nOverall Sentiment of the Sample Text:", sentiment_label)

# Forecast exchange rates for the next 341 days
try:
    forecast_days = model_fit.forecast(steps=341)
except Exception as e:
    print("Error forecasting for the next 341 days:", e)
    exit()

# Generate forecast dates for the four quarters
forecast_dates_days = pd.date_range(start=df_adjusted.index[-1] + pd.Timedelta(days=1), periods=341)

# Create a DataFrame for the forecasted exchange rates for the next 31 days
forecast_df_31days = pd.DataFrame({'Date': forecast_dates_days, 'Forecasted Exchange Rate': forecast_days})

# Filter the forecast DataFrame to include only the row for 2024-03-31
forecast_df_31days_filtered = forecast_df_31days[forecast_df_31days['Date'] == '2024-03-31']

# Filter the forecast DataFrame to include only the row for 2024-06-30
forecast_df_june_filtered = forecast_df_31days[forecast_df_31days['Date'] == '2024-06-30']

# Filter the forecast DataFrame to include only the row for 2024-09-30
forecast_df_september_filtered = forecast_df_31days[forecast_df_31days['Date'] == '2024-09-30']

# Filter the forecast DataFrame to include only the row for 2024-12-31
forecast_df_december_filtered = forecast_df_31days[forecast_df_31days['Date'] == '2024-12-31']

# Print the filtered forecasted exchange rate for 2024-03-31
print("\nForecasted Exchange Rate for end of March:")
print(forecast_df_31days_filtered.to_string(index=False))

# Print the filtered forecasted exchange rate for 2024-06-30
print("\nForecasted Exchange Rate for end of June:")
print(forecast_df_june_filtered.to_string(index=False))

# Print the filtered forecasted exchange rate for 2024-09-30
print("\nForecasted Exchange Rate for end of September:")
print(forecast_df_september_filtered.to_string(index=False))

# Print the filtered forecasted exchange rate for 2024-12-31
print("\nForecasted Exchange Rate for end of December:")
print(forecast_df_december_filtered.to_string(index=False))

# Calculate Error Rate, Accuracy, Success Rate, and Data Loss after training as percentages
error_rate = model_fit.mse / df_adjusted.mean() * 100
accuracy = 100 - error_rate
success_rate = 100 - model_fit.mse
data_loss = (model_fit.mse / df_adjusted.mean()) * 100

print("\nError Rate:", error_rate)
print("Accuracy:", accuracy)
print("Success Rate:", success_rate)
print("Data Loss:", data_loss)
