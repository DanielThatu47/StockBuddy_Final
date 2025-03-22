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
        # Generate a truly unique task ID
        timestamp = int(time.time() * 1000)  # Millisecond precision
        random_suffix = os.urandom(4).hex()  # Add random suffix
        self.task_id = f"{user_id}_{symbol}_{timestamp}_{random_suffix}"

    def run(self):
        self.thread = threading.Thread(target=self._run_prediction)
        self.thread.start()
        return self.task_id

    def _run_prediction(self):
        try:
            self.status = "running"
            self.progress = 10
            
            # Fetch stock data
            data = stock_model.fetch_stock_data(self.symbol)
            if data is None:
                self.status = "failed"
                self.result = {"error": f"Could not fetch data for {self.symbol}"}
                return
            
            self.progress = 20
            if self.stop_requested:
                self.status = "stopped"
                return
                
            # Fetch and analyze sentiment
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
            
            self.progress = 40
            if self.stop_requested:
                self.status = "stopped"
                return
                
            # Preprocess data for prediction
            scaled_data, scaler = stock_model.preprocess_data(data)
            time_step = 100
            X, y = stock_model.create_sequences(scaled_data, time_step)
            
            self.progress = 50
            if self.stop_requested:
                self.status = "stopped"
                return
                
            # Train models
            train_size = int(len(X) * 0.8)
            X_train, y_train = X[:train_size], y[:train_size]
            lstm_model = stock_model.train_lstm(X_train, y_train, time_step)
            
            self.progress = 70
            if self.stop_requested:
                self.status = "stopped"
                return
                
            # Create residuals for XGBoost
            X_train_reshaped = X_train.reshape(X_train.shape[0], time_step, 1)
            lstm_preds = lstm_model.predict(X_train_reshaped)
            residuals = y_train - lstm_preds.flatten()
            xgb_model = stock_model.train_xgboost(X_train, residuals)
            
            self.progress = 85
            if self.stop_requested:
                self.status = "stopped"
                return
                
            # Make predictions
            predictions = stock_model.predict_stock_price(lstm_model, xgb_model, scaled_data, scaler, time_step, self.days_ahead)
            
            self.progress = 95
            if self.stop_requested:
                self.status = "stopped"
                return
                
            # Create prediction results
            future_dates = []
            last_date = data.index[-1]
            for i in range(1, self.days_ahead + 1):
                next_date = last_date + timedelta(days=i)
                # Skip weekends
                while next_date.weekday() > 4:  # 5 is Saturday, 6 is Sunday
                    next_date = next_date + timedelta(days=1)
                future_dates.append(next_date.strftime("%Y-%m-%d"))
            
            prediction_data = []
            for i in range(len(predictions)):
                prediction_data.append({
                    "date": future_dates[i],
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
    task = prediction_tasks.get(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
    
    return jsonify({
        "taskId": task_id,
        "status": task.status,
        "progress": task.progress,
        "result": task.result if task.status == "completed" else None
    })

@app.route('/api/predict/stop/<task_id>', methods=['POST'])
def stop_prediction(task_id):
    task = prediction_tasks.get(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
    
    task.stop_requested = True
    return jsonify({
        "taskId": task_id,
        "status": "stop_requested"
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