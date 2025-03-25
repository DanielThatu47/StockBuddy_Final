from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import os
import threading
import time
from datetime import datetime, timedelta
import json
import model as stock_model

app = Flask(__name__)
CORS(app)

# Dictionary to store running prediction tasks
prediction_tasks = {}

class PredictionTask:
    def __init__(self, user_id, symbol, days_ahead):
        self.user_id = user_id
        self.symbol = symbol
        self.days_ahead = days_ahead
        self.progress = 0
        self.status = "pending"
        self.result = None
        self.sentiment_result = None
        self.thread = None
        self.stop_requested = False
        self.stop_acknowledged = False
        # Generate a truly unique task ID
        timestamp = int(time.time() * 1000)  # Millisecond precision
        random_suffix = os.urandom(4).hex()  # Add random suffix
        self.task_id = f"{user_id}_{symbol}_{timestamp}_{random_suffix}"

    def run(self):
        self.thread = threading.Thread(target=self._run_prediction)
        self.thread.daemon = True  # Make thread a daemon so it exits when main thread does
        self.thread.start()
        return self.task_id
    
    def is_stop_requested(self):
        """Callback function to check if stop has been requested"""
        if self.stop_requested and not self.stop_acknowledged:
            self.stop_acknowledged = True
            self.status = "stopped"
            return True
        return self.stop_requested

    def _run_prediction(self):
        try:
            self.status = "running"
            self.progress = 10
            
            # Fetch stock data (using compact data for faster processing)
            print(f"Fetching historical data for {self.symbol}...")
            data = stock_model.fetch_stock_data(self.symbol, outputsize="compact")
            if data is None:
                self.status = "failed"
                self.result = {"error": f"Could not fetch data for {self.symbol}"}
                return
            
            # Check for stop request
            if self.stop_requested:
                self.status = "stopped"
                print(f"Prediction for {self.symbol} stopped after data fetch")
                return
            
            # Filter to ensure we only have enough data
            if len(data) < 35:  # Reduced from 50 days since we're using time_step=30 now
                self.status = "failed"
                self.result = {"error": f"Insufficient data available for {self.symbol}"}
                return
                
            self.progress = 20
            if self.stop_requested:
                self.status = "stopped"
                return
                
            # Fetch and analyze sentiment
            try:
                headlines = stock_model.fetch_finnhub_news(self.symbol)
                self.progress = 30
                if self.stop_requested:
                    self.status = "stopped"
                    return
                    
                sentiment_results, sentiment_totals = stock_model.analyze_sentiment(headlines)
                sentiment_summary = stock_model.generate_sentiment_summary(sentiment_totals, headlines, self.symbol)
                self.sentiment_result = {
                    "totals": sentiment_totals,
                    "summary": sentiment_summary
                }
            except Exception as e:
                print(f"Error in sentiment analysis: {str(e)}")
                # Continue even if sentiment fails
                self.sentiment_result = {
                    "totals": {"positive": 0, "negative": 0, "neutral": 0},
                    "summary": f"Unable to analyze sentiment for {self.symbol}: {str(e)}"
                }
            
            self.progress = 40
            if self.stop_requested:
                self.status = "stopped"
                return
                
            # Preprocess data for prediction with technical indicators
            scaled_data, scaler = stock_model.preprocess_data(data)
            # Use smaller time step (30 instead of 45)
            time_step = 30
            print(f"Creating sequences with time_step={time_step}, data shape={scaled_data.shape}")
            X, y = stock_model.create_sequences(scaled_data, time_step)
            
            # Check if we have any training data
            if len(X) == 0:
                self.status = "failed"
                self.result = {"error": f"Could not create training sequences for {self.symbol}"}
                return
                
            self.progress = 50
            if self.stop_requested:
                self.status = "stopped"
                return
                
            # Train models with improved architecture
            # Use 80% for training with more data
            train_size = int(len(X) * 0.8)
            if train_size == 0:
                self.status = "failed"
                self.result = {"error": f"Not enough data points to train model for {self.symbol}"}
                return
                
            X_train, y_train = X[:train_size], y[:train_size]
            print(f"Training LSTM with {len(X_train)} samples and {X_train.shape[2]} features")
            
            # Use improved LSTM model with stop callback
            lstm_model = stock_model.train_lstm(X_train, y_train, time_step, self.is_stop_requested)
            
            # Check if training was stopped
            if self.stop_requested:
                self.status = "stopped"
                print(f"Prediction for {self.symbol} stopped after LSTM training")
                return
                
            self.progress = 70
            if self.stop_requested:
                self.status = "stopped"
                return
                
            # Create residuals for improved XGBoost
            try:
                lstm_preds = lstm_model.predict(X_train, verbose=0).flatten()
                residuals = y_train - lstm_preds
                
                # Train XGBoost with stop callback
                xgb_model = stock_model.train_xgboost(
                    X_train.reshape(X_train.shape[0], -1), 
                    residuals, 
                    self.is_stop_requested
                )
                
                # Check if XGBoost was stopped
                if self.stop_requested or xgb_model is None:
                    self.status = "stopped"
                    print(f"Prediction for {self.symbol} stopped after XGBoost training")
                    return
            except Exception as e:
                print(f"Error in XGBoost training: {str(e)}")
                # Continue with LSTM-only predictions if XGBoost fails
                xgb_model = None
                
            self.progress = 85
            if self.stop_requested:
                self.status = "stopped"
                return
                
            # Make predictions with stop callback
            try:
                predictions = stock_model.predict_stock_price(
                    lstm_model, 
                    xgb_model, 
                    scaled_data, 
                    scaler, 
                    time_step, 
                    self.days_ahead,
                    self.is_stop_requested
                )
                
                # Check if predictions were stopped
                if self.stop_requested or predictions is None:
                    self.status = "stopped"
                    print(f"Prediction for {self.symbol} stopped during prediction generation")
                    return
            except Exception as e:
                print(f"Error making predictions: {str(e)}")
                self.status = "failed"
                self.result = {"error": f"Failed to generate predictions: {str(e)}"}
                return
            
            self.progress = 95
            if self.stop_requested:
                self.status = "stopped"
                return
                
            # Create prediction results with proper business day handling
            future_dates = []
            last_date = data.index[-1]
            
            # Generate proper future dates (trading days only)
            for i in range(1, self.days_ahead + 1):
                if self.stop_requested:
                    break
                next_date = last_date + timedelta(days=i)
                # Skip weekends
                while next_date.weekday() > 4:  # 5=Saturday, 6=Sunday
                    next_date = next_date + timedelta(days=1)
                future_dates.append(next_date)
            
            # If stopped, exit early
            if self.stop_requested:
                self.status = "stopped"
                return
                
            # Remove any duplicate dates
            unique_future_dates = []
            unique_date_strs = set()
            
            for date in future_dates:
                date_str = date.strftime("%Y-%m-%d")
                if date_str not in unique_date_strs:
                    unique_date_strs.add(date_str)
                    unique_future_dates.append(date)
            
            # Ensure we have enough dates, add more if needed
            while len(unique_future_dates) < len(predictions) and not self.stop_requested:
                next_date = unique_future_dates[-1] + timedelta(days=1)
                while next_date.weekday() > 4:  # Skip weekends
                    next_date = next_date + timedelta(days=1)
                if next_date.strftime("%Y-%m-%d") not in unique_date_strs:
                    unique_future_dates.append(next_date)
                    unique_date_strs.add(next_date.strftime("%Y-%m-%d"))
            
            # If stopped, exit early
            if self.stop_requested:
                self.status = "stopped"
                return
                
            # Take only what we need
            unique_future_dates = unique_future_dates[:len(predictions)]
            
            # Create the prediction data with formatted dates
            prediction_data = []
            for i in range(min(len(unique_future_dates), len(predictions))):
                prediction_data.append({
                    "date": unique_future_dates[i].strftime("%Y-%m-%d"),
                    "price": float(predictions[i][0])
                })
            
            self.result = {
                "symbol": self.symbol,
                "predictions": prediction_data,
                "sentiment": self.sentiment_result
            }
            self.progress = 100
            self.status = "completed"
            
        except Exception as e:
            self.status = "failed"
            self.result = {"error": str(e)}
            print(f"Error in prediction: {str(e)}")

@app.route('/api/predict', methods=['POST'])
def start_prediction():
    data = request.json
    user_id = data.get('userId')
    symbol = data.get('symbol')
    days_ahead = int(data.get('daysAhead', 5))
    
    if not user_id or not symbol:
        return jsonify({"error": "Missing required parameters"}), 400
    
    task = PredictionTask(user_id, symbol, days_ahead)
    task_id = task.run()
    prediction_tasks[task_id] = task
    
    return jsonify({
        "taskId": task_id,
        "status": "pending"
    })

@app.route('/api/predict/status/<task_id>', methods=['GET'])
def prediction_status(task_id):
    try:
        task = prediction_tasks.get(task_id)
        if not task:
            return jsonify({"error": "Task not found"}), 404
        
        try:
            # Ensure result is valid JSON-serializable before returning
            if task.status == "completed" and task.result:
                # Validate result structure
                if isinstance(task.result, dict):
                    # Ensure predictions are properly formatted
                    if "predictions" in task.result and isinstance(task.result["predictions"], list):
                        # Make sure each prediction is valid
                        for pred in task.result["predictions"]:
                            if not isinstance(pred, dict) or "date" not in pred or "price" not in pred:
                                # Fix malformed prediction
                                print(f"Found malformed prediction: {pred}")
                                task.status = "failed"
                                task.result = {"error": "Malformed prediction data"}
                                break
                    else:
                        # Missing predictions
                        task.status = "failed"
                        task.result = {"error": "Missing prediction data"}
                else:
                    # Invalid result type
                    task.status = "failed"
                    task.result = {"error": "Invalid result format"}
            
            return jsonify({
                "taskId": task_id,
                "status": task.status,
                "progress": task.progress,
                "result": task.result if task.status == "completed" else None
            })
        except Exception as e:
            print(f"Error generating prediction status response: {str(e)}")
            # Return a simplified response that won't cause JSON serialization issues
            return jsonify({
                "taskId": task_id,
                "status": "error",
                "progress": task.progress,
                "error": str(e)
            })
    except Exception as e:
        print(f"Critical error in prediction status: {str(e)}")
        return jsonify({
            "taskId": task_id,
            "status": "error",
            "error": "Server error"
        }), 500

@app.route('/api/predict/stop/<task_id>', methods=['POST'])
def stop_prediction(task_id):
    task = prediction_tasks.get(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
    
    # Set the stop flag
    task.stop_requested = True
    
    # If task is running, make sure it gets stopped
    if task.thread and task.thread.is_alive():
        # Set the status immediately so client knows it's being stopped
        task.status = "stopping"
        
        # Log the stop request
        print(f"Stop requested for task {task_id} ({task.symbol})")
        
        # Wait up to 2 seconds to see if the task acknowledges the stop request
        stop_wait_start = time.time()
        while time.time() - stop_wait_start < 2:
            if task.stop_acknowledged:
                task.status = "stopped"
                break
            time.sleep(0.1)
    else:
        # If thread isn't running, just mark as stopped
        task.status = "stopped"
    
    # Return detailed status
    return jsonify({
        "taskId": task_id,
        "status": task.status,
        "symbol": task.symbol,
        "progress": task.progress,
        "stopRequested": task.stop_requested,
        "stopAcknowledged": task.stop_acknowledged
    })

@app.route('/api/predict/sentiment/<symbol>', methods=['GET'])
def get_sentiment(symbol):
    try:
        headlines = stock_model.fetch_finnhub_news(symbol)
        sentiment_results, sentiment_totals = stock_model.analyze_sentiment(headlines)
        sentiment_summary = stock_model.generate_sentiment_summary(sentiment_totals, headlines, symbol)
        
        return jsonify({
            "symbol": symbol,
            "sentiment": {
                "totals": sentiment_totals,
                "summary": sentiment_summary
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True) 