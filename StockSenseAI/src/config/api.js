export const NEWS_API_KEY = '5d617ab43bec4b4b9529c65c582eabde'; // Replace with your actual NewsAPI key
export const NEWS_API_BASE_URL = 'https://newsapi.org/v2';
export const MODEL_BACKEND_URL = 'http://192.168.252.238:5001';
export const API_URL = 'http://192.168.252.238:5000';

export const NEWS_CATEGORIES = {
  all: {
    id: 'all',
    label: 'All News',
    icon: 'newspaper',
    query: 'finance',
  },
  market: {
    id: 'market',
    label: 'Market',
    icon: 'trending-up',
    query: 'stock market',
  },
  economy: {
    id: 'economy',
    label: 'Economy',
    icon: 'cash',
    query: 'economy',
  },
  tech: {
    id: 'tech',
    label: 'Tech',
    icon: 'hardware-chip',
    query: 'technology',
  },
  crypto: {
    id: 'crypto',
    label: 'Crypto',
    icon: 'logo-bitcoin',
    query: 'cryptocurrency',
  },
};

export const getNewsApiUrl = (query, sortBy = 'publishedAt') => {
  return `${NEWS_API_BASE_URL}/everything?q=${encodeURIComponent(query)}&language=en&sortBy=${sortBy}&apiKey=${NEWS_API_KEY}`;
}; 


