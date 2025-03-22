#!/bin/bash

# Activate virtual environment if it exists
if [ -d "env" ]; then
    echo "Activating virtual environment..."
    source env/bin/activate
else
    echo "Virtual environment not found. Creating one..."
    python -m venv env
    source env/bin/activate
    pip install -r requirements.txt
fi

# Start the Flask app
echo "Starting Flask API server..."
export FLASK_APP=app.py
export FLASK_ENV=development
flask run --host=0.0.0.0 --port=5001 