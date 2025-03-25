# StockBuddy Mobile

## Project Overview
StockBuddy is a comprehensive stock prediction application that leverages AI and machine learning to provide real-time stock predictions and analysis. The project consists of three main components:

1. **Mobile App (StockSenseAI)**: A React Native/Expo mobile application that provides the user interface for viewing stock predictions, charts, and managing user profiles.

2. **Web Server (Backend)**: A Node.js/Express backend that handles user authentication, profile management, and communicates with the machine learning backend.

3. **Machine Learning Backend (Model_Backend)**: A Flask-based Python server that runs the stock prediction models and sentiment analysis.

## Features

- Real-time stock predictions using multiple ML models
- Historical stock data visualization with interactive charts
- Sentiment analysis for stocks based on news and social media
- User authentication with biometric option
- Profile management with customizable watchlists
- Multiple prediction timeframes (short-term, medium-term, long-term)

## Architecture

```
StockBuddy
├── StockSenseAI/         # Mobile application (React Native/Expo)
├── Backend/              # Web server (Node.js/Express)
└── Model_Backend/        # ML prediction server (Python/Flask)
```

## Installation & Setup

### Prerequisites
- Node.js 16+
- MongoDB
- Python 3.8+
- Expo CLI

### 1. Mobile App (StockSenseAI)

```bash
# Navigate to the mobile app directory
cd StockSenseAI

# Install dependencies
npm install

# Create .env file with configuration
# Example:
# API_BASE_URL=http://your-backend-url:5000
# ML_API_URL=http://your-ml-backend-url:5001

# Start the Expo development server
npm start
```

### 2. Web Server (Backend)

```bash
# Navigate to the backend directory
cd Backend

# Install dependencies
npm install

# Create .env file with configuration
# Example:
# PORT=5000
# MONGODB_URI=mongodb://localhost:27017/stockbuddy
# JWT_SECRET=your_jwt_secret
# NODE_ENV=development

# Start the server
npm run dev
```

### 3. Machine Learning Backend (Model_Backend)

```bash
# Navigate to the ML backend directory
cd Model_Backend

# Create a virtual environment
python -m venv env

# Activate the virtual environment
# Windows:
env\Scripts\activate
# macOS/Linux:
source env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the Flask server
python app.py
```

Alternatively, you can use the start script:
```bash
# Make the script executable (Linux/macOS)
chmod +x start.sh

# Run the script
./start.sh
```

## Development and Production

For development, you can run each component separately. For production, consider using Docker or a similar containerization solution to manage all three services.

<<<<<<< HEAD
=======
## License
>>>>>>> bebc2e9 (hello)

