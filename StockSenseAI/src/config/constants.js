<<<<<<< HEAD
export const API_URL = 'http://192.168.91.238:5000';
export const MODEL_BACKEND_URL = 'http://192.168.91.238:5001';
=======
export const API_URL = 'http://192.168.6.97:5000';
export const MODEL_BACKEND_URL = 'http://192.168.6.97:5001';
>>>>>>> 7f8ec82238fb632c29669df0f0685d99d48abf13
export const FINNHUB_API_KEY = process.env.REACT_APP_FINNHUB_API_KEY;
export const ALPHA_VANTAGE_BASE_URL = 'https://www.alphavantage.co/query';
export const ALPHA_VANTAGE_API_KEY = process.env.REACT_APP_ALPHA_VANTAGE_API_KEY;

export const STOCK_EXCHANGES = [
  { id: 'NASDAQ', name: 'NASDAQ' },
  { id: 'NYSE', name: 'New York Stock Exchange' },
  { id: 'BSE', name: 'Bombay Stock Exchange' },
  { id: 'NSE', name: 'National Stock Exchange' }
];

export const TIMEFRAMES = [
  { id: '1D', label: '1D', days: 1, interval: '5' },
  { id: '1W', label: '1W', days: 7, interval: '30' },
  { id: '1M', label: '1M', days: 30, interval: 'D' },
  { id: '3M', label: '3M', days: 90, interval: 'D' },
  { id: '1Y', label: '1Y', days: 365, interval: 'W' }
]; 