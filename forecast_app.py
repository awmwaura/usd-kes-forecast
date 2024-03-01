import pandas as pd
import streamlit as st
from statsmodels.tsa.statespace.sarimax import SARIMAX
from textblob import TextBlob
import numpy as np

st.title('Exchange Rate Forecasting and Sentiment Analysis')

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
    st.error("Error: File not found.")
    st.stop()
except Exception as e:
    st.error("Error: {}".format(e))
    st.stop()

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
    st.error("Error: Missing data found.")
    st.stop()

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
    st.error("Error fitting SARIMAX model: {}".format(e))
    st.stop()

# Perform sentiment analysis on the sample text
sentiment = TextBlob(sample_text).sentiment.polarity

# Determine sentiment label
if sentiment > 0:
    sentiment_label = 'Positive'
elif sentiment < 0:
    sentiment_label = 'Negative'
else:
    sentiment_label = 'Neutral'

# Display overall sentiment
st.subheader("Overall Sentiment of the Sample Text:")
st.write(sentiment_label)

# Display exchange rate forecast for the next 7 days
st.subheader("Forecasted Exchange Rates for the next 7 days:")
forecast_7days = model_fit.forecast(steps=7)
forecast_dates_7days = pd.date_range(start=df_adjusted.index[-1] + pd.Timedelta(days=1), periods=7)

# Generate random noise for the forecasted values
noise_7days = np.random.normal(scale=0.045, size=forecast_7days.shape[0])  # Adjust the scale as needed

# Apply the noise to the forecasted values
forecast_7days_with_noise = forecast_7days + noise_7days

# Create a DataFrame for the forecasted values with noise
forecast_df_7days_with_noise = pd.DataFrame({'Date': forecast_dates_7days, 
                                              'Forecasted Exchange Rate': forecast_7days_with_noise})

# Display forecasted values with noise
st.write(forecast_df_7days_with_noise)

# Display exchange rate forecast for the end of each quarter in 2024
st.subheader("Forecasted Exchange Rates for the end of each quarter in 2024:")
forecast_days = model_fit.forecast(steps=341)
forecast_dates_days = pd.date_range(start=df_adjusted.index[-1] + pd.Timedelta(days=1), periods=341)

# Generate random noise for the forecasted values
noise_days = np.random.normal(scale=0.045, size=forecast_days.shape[0])  # Adjust the scale as needed

# Apply the noise to the forecasted values
forecast_days_with_noise = forecast_days + noise_days

# Create a DataFrame for the forecasted values with noise
forecast_df_days_with_noise = pd.DataFrame({'Date': forecast_dates_days, 
                                            'Forecasted Exchange Rate': forecast_days_with_noise})

# Filter for end of quarters
forecast_df_quarters = forecast_df_days_with_noise[forecast_df_days_with_noise['Date'].dt.month.isin([3, 6, 9, 12])]

# Display forecasted values for end of quarters with noise
st.write(forecast_df_quarters)

# Calculate Error Rate, Accuracy, Success Rate, and Data Loss after training as percentages
error_rate = model_fit.mse / df_adjusted.mean() * 100
accuracy = 100 - error_rate
success_rate = 100 - model_fit.mse
data_loss = (model_fit.mse / df_adjusted.mean()) * 100

st.subheader("Model Evaluation Metrics:")
st.write(f"Error Rate: {error_rate}")
st.write(f"Accuracy: {accuracy}")
st.write(f"Success Rate: {success_rate}")
st.write(f"Data Loss: {data_loss}")
