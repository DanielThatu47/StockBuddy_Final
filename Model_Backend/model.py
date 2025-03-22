import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import xgboost as xgb
import plotly.graph_objects as go
from datetime import datetime, timedelta
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline

# Download VADER's lexicon (if not already downloaded)
nltk.download("vader_lexicon")

# Initialize the summarization pipeline
summarizer = pipeline("summarization")

# =============================================================================
#                         API Keys (Replace with your own keys)
# =============================================================================
ALPHAVANTAGE_API_KEY = "FI6E01DMUEW2VSWW"  # Replace with your actual Alpha Vantage API key
FINNHUB_API_KEY = "cu5gvghr01qqj8u6iau0cu5gvghr01qqj8u6iaug"  # Replace with your actual Finnhub API key

# =============================================================================
#                     STOCK PRICE PREDICTION FUNCTIONS
# =============================================================================

def fetch_stock_data(symbol, outputsize="full"):
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": ALPHAVANTAGE_API_KEY,
        "outputsize": outputsize,
        "datatype": "json"
    }
    response = requests.get(url, params=params)
    data = response.json()
    if "Time Series (Daily)" not in data:
        print(f"Error fetching data for {symbol}. Check the API key or symbol.")
        return None
    ts = data["Time Series (Daily)"]
    df = pd.DataFrame.from_dict(ts, orient="index")
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df = df.rename(columns={"4. close": "Close"})
    df["Close"] = df["Close"].astype(float)
    return df[["Close"]]

def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def create_sequences(data, time_step=100):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

def train_lstm(X_train, y_train, time_step=100):
    X_train = X_train.reshape(X_train.shape[0], time_step, 1)
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1)
    return model

def train_xgboost(X_train, residuals):
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.01)
    xgb_model.fit(X_train, residuals)
    return xgb_model

def predict_stock_price(lstm_model, xgb_model, data, scaler, time_step=100, days_ahead=5):
    temp_input = data[-time_step:].tolist()
    predictions = []
    for _ in range(days_ahead):
        lstm_input = np.array(temp_input[-time_step:]).reshape(1, time_step, 1)
        lstm_pred = lstm_model.predict(lstm_input)[0][0]
        xgb_input = np.array(temp_input[-time_step:]).reshape(1, -1)
        xgb_pred = xgb_model.predict(xgb_input)[0]
        combined_pred = lstm_pred + xgb_pred
        predictions.append(combined_pred)
        temp_input.append([combined_pred])
    final_predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return final_predictions

def plot_prices(data, predictions, symbol, days_ahead):
    fig = go.Figure()

    # Use data from the last 3 months for actual prices
    three_months_ago = data.index[-1] - pd.DateOffset(months=3)
    actual_data = data.loc[three_months_ago:]

    # Prepare future dates for predicted prices
    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=days_ahead)
    combined_dates = np.concatenate([actual_data.index, future_dates])
    combined_prices = np.concatenate([actual_data['Close'], predictions.flatten()])

    fig.add_trace(go.Scatter(
        x=combined_dates,
        y=combined_prices,
        mode='lines',
        name='Predicted Price',
        hovertemplate='Date: %{x}<br>Price: $%{y:.2f}',
        line=dict(color='orange')
    ))

    fig.add_trace(go.Scatter(
        x=actual_data.index,
        y=actual_data['Close'],
        mode='lines',
        name='Actual Price',
        hovertemplate='Date: %{x}<br>Price: $%{y:.2f}',
        line=dict(color='blue')
    ))

    fig.update_layout(
        title=f'Stock Price Prediction for {symbol}',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        template='plotly_dark',
        hovermode='x unified'
    )

    fig.show()

# =============================================================================
#                   NEWS SENTIMENT ANALYSIS FUNCTIONS
# =============================================================================

def fetch_finnhub_news(company_symbol):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=28)
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    url = f"https://finnhub.io/api/v1/company-news?symbol={company_symbol}&from={start_date_str}&to={end_date_str}&token={FINNHUB_API_KEY}"
    response = requests.get(url)

    if response.status_code == 200:
        articles = response.json()
        headlines = [article["headline"] for article in articles if "headline" in article]
        return headlines
    else:
        print(f"Error fetching news: {response.status_code}, Message: {response.json()}")
        return []

def analyze_sentiment(headlines):
    sid = SentimentIntensityAnalyzer()
    sentiment_results = []
    sentiment_totals = {"positive": 0, "negative": 0, "neutral": 0}

    for headline in headlines:
        sentiment = sid.polarity_scores(headline)
        sentiment_results.append({"headline": headline, "sentiment": sentiment})

        if sentiment["compound"] > 0.05:
            sentiment_totals["positive"] += 1
        elif sentiment["compound"] < -0.05:
            sentiment_totals["negative"] += 1
        else:
            sentiment_totals["neutral"] += 1

    return sentiment_results, sentiment_totals

def plot_sentiment_pie(sentiment_totals, company_symbol):
    labels = ["Positive", "Negative", "Neutral"]
    sizes = [
        sentiment_totals["positive"],
        sentiment_totals["negative"],
        sentiment_totals["neutral"],
    ]
    colors = ["#2ecc71", "#e74c3c", "#95a5a6"]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=sizes,
        marker=dict(colors=colors, line=dict(color='white', width=0)),
        textinfo='percent+label',
        textfont_size=20,
    )])

    fig.update_layout(
        title=f"Sentiment Distribution for {company_symbol} (Last 28 Days)",
        showlegend=True,
        margin=dict(t=50, b=10, l=30, r=130)
    )

    fig.show()

# =============================================================================
#                    AI SUMMARY FUNCTIONS (Sentiment & Prediction)
# =============================================================================

def generate_sentiment_summary(sentiment_totals, headlines, company_symbol):
    summary_text = (
        f"Over the past 28 days, there have been {len(headlines)} news articles about {company_symbol}. "
        f"Sentiment analysis shows {sentiment_totals['positive']} positive articles, "
        f"{sentiment_totals['negative']} negative articles, and {sentiment_totals['neutral']} neutral articles."
    )
    if headlines:
        combined_text = " ".join(headlines[:5])
        ai_summary = summarizer(combined_text, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]
        summary_text += f" Key insights: {ai_summary}"
    return summary_text

def generate_prediction_summary(pred_df, company_symbol):
    first_price = pred_df["Predicted Price"].iloc[0]
    last_price = pred_df["Predicted Price"].iloc[-1]
    summary_text = (
        f"The predicted stock prices for {company_symbol} range from ${first_price:.2f} to ${last_price:.2f} "
        "over the forecast period."
    )
    return summary_text

# =============================================================================
#                          UNIFIED MAIN FUNCTION
# =============================================================================

def main():
    symbol = input("Enter the stock symbol (e.g., AAPL): ").upper()
    try:
        days_ahead = int(input("Enter the number of future days to predict (e.g., 1, 2, 3, 5): "))
    except ValueError:
        print("Invalid input for number of days. Please enter an integer.")
        return

    # -------------------- Stock Price Prediction -------------------- #
    data = fetch_stock_data(symbol)
    if data is None:
        return

    print("\nFetching and processing historical stock data...")
    scaled_data, scaler = preprocess_data(data)
    time_step = 100
    X, y = create_sequences(scaled_data, time_step)
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]

    print("Training LSTM model...")
    lstm_model = train_lstm(X_train, y_train, time_step)
    lstm_train_preds = lstm_model.predict(X_train.reshape(X_train.shape[0], time_step, 1)).flatten()
    residuals = y_train - lstm_train_preds

    print("Training XGBoost on LSTM residuals...")
    xgb_model = train_xgboost(X_train.reshape(X_train.shape[0], -1), residuals)

    print(f"Predicting stock prices for the next {days_ahead} days...")
    predictions = predict_stock_price(lstm_model, xgb_model, scaled_data, scaler, time_step, days_ahead)
    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=days_ahead)
    pred_df = pd.DataFrame({"Date": future_dates, "Predicted Price": predictions.flatten()})
    print("\nPredicted Prices:")
    print(pred_df)

    # Display the stock price prediction chart
    plot_prices(data, predictions, symbol, days_ahead)

    # Generate and print prediction summary
    prediction_summary = generate_prediction_summary(pred_df, symbol)
    print("\nPrediction Summary:")
    print(prediction_summary)

    # ------------------- News Sentiment Analysis -------------------- #

    headlines = fetch_finnhub_news(symbol)
    if headlines:
        sentiment_results, sentiment_totals = analyze_sentiment(headlines)
        # print("\nSentiment Analysis Results:")
        # for result in sentiment_results:
            # print(f"Headline: {result['headline']}")
            # print(f"Sentiment Scores: {result['sentiment']}")
            # print("-" * 50)

        # Display the sentiment pie chart
        plot_sentiment_pie(sentiment_totals, symbol)

        # Generate and print sentiment summary
        sentiment_summary = generate_sentiment_summary(sentiment_totals, headlines, symbol)
        print("\nSentiment Summary:")
        print(sentiment_summary)
    else:
        print("No headlines found for the specified company.")

if __name__ == "__main__":
    main()
