# main.py
from fastapi import FastAPI, Query, File, UploadFile
from fastapi.responses import JSONResponse
from routers import routes
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from textblob import TextBlob

app = FastAPI()

@app.post("/forecast/")
async def forecast_exchange_rate(
    excel_file: UploadFile = File(...),
    bill_rate: float = Query(...),
    inflation_rate: float = Query(...),
    bond_rate: float = Query(...),
    cbk_rates: float = Query(...),
    sentiment_analysis_text: str = Query(...)
):
    # Load data
    try:
        df = pd.read_excel(excel_file.file)
    except Exception as e:
        return JSONResponse(content={"error": f"Error reading Excel file: {e}"}, status_code=400)

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
        return JSONResponse(content={"error": "Missing data found."}, status_code=400)

    # Split data into train and test sets
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]

    # Fit SARIMAX model
    try:
        model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        model_fit = model.fit()
    except Exception as e:
        return JSONResponse(content={"error": f"Error fitting SARIMAX model: {e}"}, status_code=400)

    # Forecast based on historical data trends
    try:
        forecast = model_fit.forecast(steps=14)
    except Exception as e:
        return JSONResponse(content={"error": f"Error forecasting: {e}"}, status_code=400)

    # Calculate sentiment polarity
    try:
        sentiment = TextBlob(sentiment_analysis_text).sentiment.polarity
    except Exception as e:
        sentiment = 0.0

    # Incorporate other economic factors
    try:
        # Add randomness to the adjustment based on sentiment and economic factors
        np.random.seed(42)  # for reproducibility
        random_factors = np.random.normal(loc=1, scale=0.01, size=14)  # Lower randomness scale

        # Apply sentiment analysis and economic factors to adjust the forecast with randomness
        forecast_adjusted = forecast * (1 + sentiment) * (1 + inflation_rate) * (1 + bond_rate) * random_factors

        # Print forecast with month and day
        forecast_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=14)
        forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted Exchange Rate': forecast_adjusted})
        forecast_result = forecast_df.to_dict(orient='records')
    except Exception as e:
        return JSONResponse(content={"error": f"Error processing forecast: {e}"}, status_code=400)

    # Print sentiment
    sentiment_result = {"Sentiment Polarity": sentiment}

    return {"forecast": forecast_result, "sentiment": sentiment_result}

# Include routers
app.include_router(routes.router)
