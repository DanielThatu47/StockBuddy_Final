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
import time

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
    
    # Retry logic for API requests
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            # Check if we got valid data
            if "Time Series (Daily)" not in data:
                # Check for API errors
                if "Error Message" in data:
                    print(f"API Error: {data['Error Message']}")
                elif "Note" in data:
                    print(f"API Limit Note: {data['Note']}")
                else:
                    print(f"Unknown error fetching data for {symbol}. Response: {data}")
                
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    print(f"Failed to fetch data for {symbol} after {max_retries} attempts")
                    return None
            else:
                break  # Success, exit the retry loop
        except Exception as e:
            print(f"Exception during data fetch: {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds... (Attempt {attempt+1}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            else:
                print(f"Failed to fetch data for {symbol} after {max_retries} attempts due to error: {str(e)}")
                return None
    
    # If we reached here, we have valid data
    ts = data["Time Series (Daily)"]
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(ts, orient="index")
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    
    # Convert all price columns to float
    for col in ["1. open", "2. high", "3. low", "4. close", "5. volume"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
    
    # Rename columns for easier access
    df = df.rename(columns={
        "1. open": "Open",
        "2. high": "High",
        "3. low": "Low", 
        "4. close": "Close",
        "5. volume": "Volume"
    })
    
    # Verify we have the latest data
    latest_date = df.index[-1]
    today = pd.Timestamp.now().normalize()
    market_closed_days = 0
    
    # Account for weekends and typical market hours
    if today.dayofweek >= 5:  # Weekend (5=Saturday, 6=Sunday)
        market_closed_days = today.dayofweek - 4  # Days since Friday
    elif today.hour < 16:  # Before market close at 4 PM
        market_closed_days = 1  # Consider yesterday's close as latest
        
    # Calculate the expected latest trading day
    expected_latest = today - pd.Timedelta(days=market_closed_days)
    
    # Check if our data is reasonably current
    date_diff = (expected_latest - latest_date).days
    if date_diff > 5:  # More than 5 days difference is suspicious
        print(f"WARNING: Latest data for {symbol} is from {latest_date.strftime('%Y-%m-%d')}, "
              f"which is {date_diff} days before the expected latest trading day.")
    
    # Print first 10 days and last 10 days
    print("\nFirst 10 days of data:")
    print(df.head(10))
    print("\nLast 10 days of data:")
    print(df.tail(10))
    
    # Print the actual latest price
    print(f"\nLatest closing price for {symbol} (as of {latest_date.strftime('%Y-%m-%d')}): ${df['Close'].iloc[-1]:.2f}")
    
    # Add technical indicators
    df = add_technical_indicators(df)
    
    return df

def add_technical_indicators(df):
    """Add technical indicators to the dataframe to improve model accuracy"""
    try:
        # Make sure we have the necessary columns
        required_cols = ["Close", "Open", "High", "Low"]
        for col in required_cols:
            if col not in df.columns:
                print(f"Warning: {col} column missing, cannot calculate all indicators")
                # Return only Close if we don't have all required columns
                return df[["Close"]]
        
        # Price features
        df['PriceFeature'] = (df['Close'] + df['Open'] + df['High'] + df['Low']) / 4
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Moving Averages
        df['SMA5'] = df['Close'].rolling(window=5).mean()
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        
        # MACD
        df['EMA12'] = df['Close'].ewm(span=12).mean()
        df['EMA26'] = df['Close'].ewm(span=26).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['Signal'] = df['MACD'].ewm(span=9).mean()
        
        # Bollinger Bands
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['Upper_Band'] = df['MA20'] + (df['Close'].rolling(window=20).std() * 2)
        df['Lower_Band'] = df['MA20'] - (df['Close'].rolling(window=20).std() * 2)
        
        # Price rate of change
        df['ROC'] = df['Close'].pct_change(periods=5) * 100
        
        # Volatility indicator
        df['Volatility'] = df['Close'].rolling(window=10).std() / df['Close'] * 100
        
        # Drop NaN values
        df = df.dropna()
        
        # Select features for model input
        features = ["Close", "PriceFeature", "RSI", "SMA5", "SMA20", "MACD", "Signal", "Upper_Band", "Lower_Band", "ROC", "Volatility"]
        return df[features]
    except Exception as e:
        print(f"Error adding technical indicators: {str(e)}")
        # Return just the Close price column if calculations fail
        if "Close" in df.columns:
            return df[["Close"]]
        return df

def preprocess_data(data):
    """Preprocess the data before model training"""
    # Separate features and target
    features = data.columns
    
    # Create separate scalers for each feature to preserve relationships
    scalers = {}
    scaled_data = np.zeros((len(data), len(features)))
    
    # Scale each feature separately
    for i, feature in enumerate(features):
        scalers[feature] = MinMaxScaler(feature_range=(0, 1))
        scaled_data[:, i] = scalers[feature].fit_transform(data[feature].values.reshape(-1, 1)).flatten()
    
    # Create a master scaler for the target (Close price)
    master_scaler = scalers["Close"]
    
    return scaled_data, master_scaler

def create_sequences(data, time_step=30):
    """Create input sequences and target values"""
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        # Use all features for X
        X.append(data[i:(i + time_step), :])
        # Use only the Close price (first column) for y
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

def train_lstm(X_train, y_train, time_step=30, stop_requested_callback=None):
    """Train an optimized LSTM model for faster yet accurate predictions"""
    # Get the number of features from input shape
    n_features = X_train.shape[2]
    
    # Reshape input to be [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], time_step, n_features)
    
    # More efficient model architecture with better learning capabilities
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(time_step, n_features), 
             recurrent_dropout=0.0),  # No recurrent dropout for faster training
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    # Use Adam optimizer with improved learning rate
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback
    
    # Create a custom callback that checks if stop was requested
    class StopCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            if stop_requested_callback and stop_requested_callback():
                self.model.stop_training = True
                print("Training stopped early by user request")
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error', run_eagerly=True)
    
    # Define callbacks for better training with increased patience
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=0.0001, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    
    callbacks = [reduce_lr, early_stopping]
    
    # Add stop callback if provided
    if stop_requested_callback:
        callbacks.append(StopCallback())
    
    print(f"Training LSTM model with input shape: {X_train.shape}")
    model.fit(
        X_train, y_train, 
        epochs=30,  # Increased epochs to allow for more training
        batch_size=64,  # Larger batch size for faster training
        validation_split=0.2,  # Use validation set
        callbacks=callbacks,
        verbose=1
    )
    return model

def train_xgboost(X_train, residuals, stop_requested_callback=None):
    """Train an optimized XGBoost model on LSTM residuals"""
    # Check if stop requested before starting
    if stop_requested_callback and stop_requested_callback():
        print("XGBoost training cancelled due to stop request")
        return None
        
    # Optimized parameters for faster training but maintaining accuracy
    params = {
        'objective': 'reg:squarederror',
        'n_estimators': 300,  # Increased number of estimators
        'learning_rate': 0.1,  # Adjusted learning rate
        'max_depth': 6,        # Increased depth
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,  # Helps prevent overfitting
        'gamma': 0.2,  # Minimum loss reduction for further partition
        'reg_alpha': 0.2,  # L1 regularization
        'reg_lambda': 1.2,  # L2 regularization
        'tree_method': 'hist',  # Faster algorithm
        'grow_policy': 'lossguide'  # Faster tree growing strategy
    }
    
    # New style callback implementation
    if stop_requested_callback:
        class StopCallbackHandler:
            def after_iteration(self, model, epoch, evals_log):
                if stop_requested_callback():
                    print("XGBoost training stopped by user request")
                    return False
                return True
        
        # Create the XGBoost model with callbacks in the constructor (new style)
        xgb_model = xgb.XGBRegressor(**params)
        xgb_model.fit(X_train, residuals, callbacks=[StopCallbackHandler()])
    else:
        # If no callback needed, just use normal training
        xgb_model = xgb.XGBRegressor(**params)
        xgb_model.fit(X_train, residuals, eval_metric=['rmse'], early_stopping_rounds=20, 
                     verbose=True, eval_set=[(X_train, residuals)])
    
    return xgb_model

def predict_stock_price(lstm_model, xgb_model, data, scaler, time_step=30, days_ahead=5, stop_requested_callback=None):
    """Make predictions using both LSTM and XGBoost models with proper anchoring to last actual price"""
    # Check if we should stop before starting
    if stop_requested_callback and stop_requested_callback():
        print("Prediction cancelled due to stop request")
        return None
        
    n_features = data.shape[1]
    temp_input = data[-time_step:].tolist()
    
    # Get the actual last closing price from the original data (not scaled)
    last_actual_close = scaler.inverse_transform(np.array([[data[-1, 0]]]))[0][0]
    print(f"Using last actual closing price: ${last_actual_close:.2f} as base for predictions")
    
    # Calculate historical volatility (std dev of returns) to use for realistic fluctuations
    # Get the original close prices (not scaled)
    original_prices = scaler.inverse_transform(data[:, 0].reshape(-1, 1))
    
    # Calculate daily returns
    daily_returns = np.diff(original_prices, axis=0) / original_prices[:-1]
    
    # Calculate the standard deviation of daily returns (volatility)
    volatility = np.std(daily_returns)
    print(f"Historical volatility: {volatility:.4f}")
    
    # Initialize predictions list to store results
    predictions = []
    
    # First, predict the actual last price to calibrate the model
    lstm_input = np.array(temp_input[-time_step:]).reshape(1, time_step, n_features)
    lstm_pred = lstm_model.predict(lstm_input, verbose=0)[0][0]
    
    xgb_input = np.array(temp_input[-time_step:]).reshape(1, -1)
    
    try:
        if xgb_model is not None:
            xgb_pred = xgb_model.predict(xgb_input)[0]
            combined_pred = lstm_pred + xgb_pred
        else:
            combined_pred = lstm_pred
    except Exception as e:
        print(f"Error in XGBoost prediction: {str(e)}")
        combined_pred = lstm_pred
    
    # Get the model's prediction for the current price
    model_predicted_current = scaler.inverse_transform(np.array([[combined_pred]]))[0][0]
    
    # Calculate correction factor to adjust predictions to actual price
    correction_factor = last_actual_close / model_predicted_current if model_predicted_current > 0 else 1.0
    print(f"Model calibration: Model predicted ${model_predicted_current:.2f} for last known price")
    print(f"Correction factor: {correction_factor:.4f}")
    
    # Store the previous day's prediction for flux calculation
    prev_day_pred = combined_pred
    
    # Generate predictions for future days
    for day in range(days_ahead):
        # Check for stop request
        if stop_requested_callback and stop_requested_callback():
            print(f"Prediction stopped at day {day}/{days_ahead}")
            break
            
        # Prepare input for LSTM (all features)
        lstm_input = np.array(temp_input[-time_step:]).reshape(1, time_step, n_features)
        
        # Get LSTM prediction (for Close price only)
        lstm_pred = lstm_model.predict(lstm_input, verbose=0)[0][0]
        
        # Prepare input for XGBoost
        xgb_input = np.array(temp_input[-time_step:]).reshape(1, -1)
        
        # Get XGBoost prediction (adjustment to LSTM prediction)
        try:
            if xgb_model is not None:
                xgb_pred = xgb_model.predict(xgb_input)[0]
                # Combine predictions
                combined_pred = lstm_pred + xgb_pred
            else:
                # If XGBoost was stopped, just use LSTM prediction
                combined_pred = lstm_pred
        except Exception as e:
            print(f"Error in XGBoost prediction: {str(e)}")
            combined_pred = lstm_pred  # Fallback to LSTM if XGBoost fails
        
        # Add some natural variation to the prediction based on historical volatility
        # This prevents straight-line predictions by introducing realistic flux
        
        # Base flux on historical volatility and a random factor
        # Get unscaled values for better flux calculation
        prev_unscaled = scaler.inverse_transform(np.array([[prev_day_pred]]))[0][0]
        current_unscaled = scaler.inverse_transform(np.array([[combined_pred]]))[0][0]
        
        # Calculate the trend direction and magnitude
        price_change = current_unscaled - prev_unscaled
        trend_direction = 1 if price_change >= 0 else -1
        
        # Generate realistic flux - more volatility for later days (compound effect)
        day_volatility = volatility * (1 + day * 0.1)  # Reduced volatility scaling for later days
        
        # Apply variation - adjust volatility to a realistic range
        adjusted_volatility = min(day_volatility, 0.015)  # Reduced maximum volatility cap to 1.5%
        
        # Add some random variation that follows historical patterns but with direction bias
        random_factor = np.random.normal(0, adjusted_volatility)
        
        # Bias the random factor based on trend direction
        # For upward trends (good): be more conservative with upward flux and more generous with downward flux
        # For downward trends (bad): be more pessimistic with downward flux and more conservative with upward flux
        if trend_direction > 0:  # Upward trend
            if np.random.random() < 0.7:  # 70% chance of following trend
                # Follow upward trend but conservatively
                flux_factor = abs(random_factor) * trend_direction * 0.15  # Reduced upward flux
            else:
                # Stronger counter-trend for upward trends
                flux_factor = -abs(random_factor) * trend_direction * 0.3  # Stronger downward flux in uptrend
        else:  # Downward trend
            if np.random.random() < 0.8:  # 80% chance of following trend
                # Follow downward trend more strongly
                flux_factor = abs(random_factor) * trend_direction * 0.25  # Stronger downward flux
            else:
                # Weaker counter-trend for downward trends
                flux_factor = -abs(random_factor) * trend_direction * 0.1  # Weaker upward flux in downtrend
            
        # Calculate flux amount
        flux_amount = prev_unscaled * flux_factor
        
        # Apply flux in original scale
        adjusted_unscaled = current_unscaled + flux_amount
        
        # Convert back to scaled value
        adjusted_pred = scaler.transform(np.array([[adjusted_unscaled]]))[0][0]
        
        # Create the next day's features with the adjusted prediction
        next_row = temp_input[-1].copy()
        next_row[0] = adjusted_pred  # Update Close price
        
        # Store for next day's calculation
        prev_day_pred = adjusted_pred
        
        # Add the adjusted prediction
        predictions.append(adjusted_pred)
        temp_input.append(next_row)
    
    # If we have no predictions (all steps were stopped), return None
    if not predictions:
        return None
        
    # Convert predictions back to original scale
    final_predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    # Apply correction factor to align with actual last closing price
    corrected_predictions = final_predictions * correction_factor
    
    print("\nOriginal vs. Corrected Predictions:")
    for i in range(len(final_predictions)):
        print(f"Day {i+1}: Original ${final_predictions[i][0]:.2f} -> Corrected ${corrected_predictions[i][0]:.2f}")
    
    return corrected_predictions

def plot_prices(data, predictions, symbol, days_ahead):
    """Generate improved plot with proper date handling"""
    fig = go.Figure()

    # For display, use only the last 3 months of actual data
    three_months_ago = data.index[-1] - pd.DateOffset(months=3)
    actual_data = data.loc[three_months_ago:]
    
    # Get the 'Close' column for actual data
    if isinstance(actual_data, pd.DataFrame) and 'Close' in actual_data.columns:
        close_prices = actual_data['Close']
    else:
        close_prices = actual_data.iloc[:, 0]  # Assume first column is Close

    # Generate future dates, skipping weekends
    future_dates = []
    last_date = data.index[-1]
    for i in range(1, days_ahead + 1):
        next_date = last_date + timedelta(days=i)
        # Skip weekends
        while next_date.weekday() > 4:  # 5 is Saturday, 6 is Sunday
            next_date = next_date + timedelta(days=1)
        future_dates.append(next_date)
    
    # Ensure unique dates (no duplicates)
    future_dates = list(dict.fromkeys(future_dates))
    
    # If we have fewer dates than predictions (due to deduplication), use only available dates
    prediction_data = predictions[:len(future_dates)].flatten()
    
    # Plot the predictions (future prices)
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=prediction_data,
        mode='lines+markers',
        name='Predicted Price',
        hovertemplate='Date: %{x|%Y-%m-%d}<br>Price: $%{y:.2f}',
        line=dict(color='orange', width=3)
    ))

    # Plot the actual data (historical prices)
    fig.add_trace(go.Scatter(
        x=close_prices.index,
        y=close_prices.values,
        mode='lines',
        name='Actual Price',
        hovertemplate='Date: %{x|%Y-%m-%d}<br>Price: $%{y:.2f}',
        line=dict(color='blue', width=2)
    ))

    # Highlight the latest actual price point
    fig.add_trace(go.Scatter(
        x=[close_prices.index[-1]],
        y=[close_prices.values[-1]],
        mode='markers',
        name='Latest Price',
        marker=dict(color='green', size=10, symbol='circle'),
        hovertemplate='Date: %{x|%Y-%m-%d}<br>Price: $%{y:.2f}'
    ))

    # Improve the layout
    fig.update_layout(
        title=f'Stock Price Prediction for {symbol}',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        xaxis=dict(
            type='date',
            tickformat='%Y-%m-%d',
            tickmode='auto',
            nticks=10,
            showgrid=True
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(230, 230, 230, 0.5)'
        ),
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.show()

# =============================================================================
#                   NEWS SENTIMENT ANALYSIS FUNCTIONS
# =============================================================================

def fetch_finnhub_news(company_symbol):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=28)  # Increased from 14 days to 28 days
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    url = f"https://finnhub.io/api/v1/company-news?symbol={company_symbol}&from={start_date_str}&to={end_date_str}&token={FINNHUB_API_KEY}"
    response = requests.get(url)

    try:
        if response.status_code == 200:
            articles = response.json()
            headlines = [article["headline"] for article in articles if "headline" in article]
            return headlines
        else:
            print(f"Error fetching news: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error parsing news response: {str(e)}")
        return []

def analyze_sentiment(headlines):
    try:
        sid = SentimentIntensityAnalyzer()
        sentiment_results = []
        sentiment_totals = {"positive": 0, "negative": 0, "neutral": 0}

        for headline in headlines:
            if not headline or not isinstance(headline, str):
                continue
                
            sentiment = sid.polarity_scores(headline)
            sentiment_results.append({"headline": headline, "sentiment": sentiment})

            if sentiment["compound"] > 0.05:
                sentiment_totals["positive"] += 1
            elif sentiment["compound"] < -0.05:
                sentiment_totals["negative"] += 1
            else:
                sentiment_totals["neutral"] += 1

        return sentiment_results, sentiment_totals
    except Exception as e:
        print(f"Error in sentiment analysis: {str(e)}")
        return [], {"positive": 0, "negative": 0, "neutral": 0}

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
    try:
        summary_text = (
            f"Over the past 28 days, there have been {len(headlines)} news articles about {company_symbol}. "
            f"Sentiment analysis shows {sentiment_totals['positive']} positive articles, "
            f"{sentiment_totals['negative']} negative articles, and {sentiment_totals['neutral']} neutral articles."
        )
        if headlines and len(headlines) >= 3:
            try:
                combined_text = " ".join(headlines[:3])
                ai_summary = summarizer(combined_text, max_length=100, min_length=30, do_sample=False)[0]["summary_text"]
                summary_text += f" Key insights: {ai_summary}"
            except Exception as e:
                print(f"Error generating AI summary: {str(e)}")
                # Fallback to simple summary if AI summarization fails
                if len(headlines) > 0:
                    summary_text += f" Most recent headline: {headlines[0]}"
        return summary_text
    except Exception as e:
        print(f"Error in generate_sentiment_summary: {str(e)}")
        return f"Unable to generate sentiment summary for {company_symbol}."

def generate_prediction_summary(pred_df, company_symbol):
    first_price = pred_df["Predicted Price"].iloc[0]
    last_price = pred_df["Predicted Price"].iloc[-1]
    summary_text = (
        f"The predicted stock prices for {company_symbol} range from ${first_price:.2f} to ${last_price:.2f} "
        "over the forecast period."
    )
    return summary_text

def display_price_table(data, predictions, symbol, days_ahead):
    """Display prediction results as a table instead of a graph"""
    # Get the last actual closing price
    if isinstance(data, pd.DataFrame) and 'Close' in data.columns:
        last_price = data['Close'].iloc[-1]
        last_date = data.index[-1]
    else:
        last_price = data.iloc[-1, 0]  # Assume first column is Close
        last_date = data.index[-1]
    
    # Generate future dates, skipping weekends
    future_dates = []
    
    for i in range(1, days_ahead + 1):
        next_date = last_date + timedelta(days=i)
        # Skip weekends
        while next_date.weekday() > 4:  # 5 is Saturday, 6 is Sunday
            next_date = next_date + timedelta(days=1)
        future_dates.append(next_date)
    
    # Ensure unique dates (no duplicates)
    future_dates = list(dict.fromkeys(future_dates))
    
    # If we have fewer dates than predictions (due to deduplication), use only available dates
    prediction_data = predictions[:len(future_dates)].flatten()
    
    # Create a DataFrame for clear comparison with the last price
    last_price_row = pd.DataFrame({
        "Date": [last_date.strftime("%Y-%m-%d")],
        "Price": [f"${last_price:.2f}"],
        "Change": ["0.00%"],
        "Note": ["Actual last closing price"]
    })
    
    # Create the prediction DataFrame
    pred_rows = []
    for i, (date, price) in enumerate(zip(future_dates, prediction_data)):
        change_pct = ((price - last_price) / last_price) * 100
        pred_rows.append({
            "Date": date.strftime("%Y-%m-%d"),
            "Price": f"${price:.2f}",
            "Change": f"{change_pct:.2f}%",
            "Note": f"Day {i+1} prediction"
        })
    
    pred_df = pd.DataFrame(pred_rows)
    
    # Combine for display
    combined_df = pd.concat([last_price_row, pred_df], ignore_index=True)
    
    # Print the table with clear headers
    print(f"\n{symbol} Stock Price Prediction Table:")
    print("=" * 80)
    print(combined_df.to_string(index=False))
    print("=" * 80)
    
    # Create a return DataFrame with just the predictions for other functions
    result_df = pd.DataFrame({
        "Date": [date.strftime("%Y-%m-%d") for date in future_dates],
        "Predicted Price": prediction_data
    })
    
    return result_df

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
    print(f"\nFetching historical data for {symbol}...")
    data = fetch_stock_data(symbol, outputsize="full")  # Use full data
    if data is None:
        return
        
    # Check if we have enough data
    if len(data) < 65:  # We need at least 65 days for time_step=60
        print(f"Not enough data points for {symbol}. Need at least 65 days.")
        return

    print("\nProcessing historical stock data with technical indicators...")
    scaled_data, scaler = preprocess_data(data)
    
    # Use smaller time step for prediction
    time_step = 60
    print(f"Creating sequences with time_step={time_step}, data shape={scaled_data.shape}")
    X, y = create_sequences(scaled_data, time_step)
    
    if len(X) == 0:
        print(f"Could not create sequences for {symbol}. Not enough data points.")
        return
        
    # Use 80% for training
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    print(f"Training data size: {len(X_train)} samples with {X_train.shape[2]} features")

    print("Training LSTM model with technical indicators...")
    lstm_model = train_lstm(X_train, y_train, time_step)
    
    # Calculate the residuals from LSTM predictions for XGBoost
    lstm_train_preds = lstm_model.predict(X_train).flatten()
    residuals = y_train - lstm_train_preds
    
    print("Training XGBoost model on LSTM residuals...")
    # Reshape X_train to 2D for XGBoost
    xgb_model = train_xgboost(X_train.reshape(X_train.shape[0], -1), residuals)

    print(f"Predicting stock prices for the next {days_ahead} days...")
    predictions = predict_stock_price(lstm_model, xgb_model, scaled_data, scaler, time_step, days_ahead)
    
    # Generate proper future dates with business day handling
    future_dates = []
    last_date = data.index[-1]
    
    for i in range(1, days_ahead + 1):
        next_date = last_date + timedelta(days=i)
        # Skip weekends
        while next_date.weekday() > 4:  # 5=Saturday, 6=Sunday
            next_date = next_date + timedelta(days=1)
        future_dates.append(next_date)
    
    # Ensure dates are unique
    future_dates = list(dict.fromkeys(future_dates))
    
    # Create dataframe with predictions
    pred_df = pd.DataFrame({
        "Date": [date.strftime("%Y-%m-%d") for date in future_dates[:len(predictions)]],
        "Predicted Price": predictions.flatten()[:len(future_dates)]
    })
    
    print("\nPredicted Prices:")
    print(pred_df)

    # Display the prediction results as a table instead of a plot
    display_price_table(data, predictions, symbol, days_ahead)

    # Generate and print prediction summary
    prediction_summary = generate_prediction_summary(pred_df, symbol)
    print("\nPrediction Summary:")
    print(prediction_summary)

    # ------------------- News Sentiment Analysis -------------------- #
    print("\nFetching news headlines for sentiment analysis...")
    headlines = fetch_finnhub_news(symbol)
    if headlines:
        print(f"Retrieved {len(headlines)} headlines for sentiment analysis")
        sentiment_results, sentiment_totals = analyze_sentiment(headlines)

        # Display the sentiment pie chart
        plot_sentiment_pie(sentiment_totals, symbol)

        # Generate and print sentiment summary
        sentiment_summary = generate_sentiment_summary(sentiment_totals, headlines, symbol)
        print("\nSentiment Summary:")
        print(sentiment_summary)
        
        # Combine sentiment with prediction for a comprehensive outlook
        print("\nCombined Market Outlook:")
        sentiment_score = (sentiment_totals["positive"] - sentiment_totals["negative"]) / max(1, sum(sentiment_totals.values()))
        sentiment_direction = "positive" if sentiment_score > 0.2 else "negative" if sentiment_score < -0.2 else "neutral"
        price_direction = "rising" if pred_df["Predicted Price"].iloc[-1] > pred_df["Predicted Price"].iloc[0] else "falling"
        
        print(f"Technical analysis suggests {symbol} stock price is {price_direction} over the next {days_ahead} trading days.")
        print(f"Market sentiment is currently {sentiment_direction} with a score of {sentiment_score:.2f}.")
    else:
        print("No headlines found for the specified company.")

if __name__ == "__main__":
    main()
