const API_KEY = process.env.REACT_APP_FINNHUB_API_KEY; // Replace with your actual API key
const BASE_URL = 'https://finnhub.io/api/v1';

/**
 * Service for interacting with the Finnhub API
 */
const FinnhubService = {
  /**
   * Search for stocks by symbol or name
   * @param {string} query - Search term (company name or symbol)
   * @returns {Promise<Array>} - List of matching stocks
   */
  searchStocks: async (query) => {
    try {
      const response = await fetch(`${BASE_URL}/search?q=${query}&token=${API_KEY}`);
      const data = await response.json();
      return data.result || [];
    } catch (error) {
      console.error('Error searching stocks:', error);
      throw error;
    }
  },

  /**
   * Get company details by symbol
   * @param {string} symbol - Stock symbol
   * @returns {Promise<Object>} - Company details
   */
  getCompanyProfile: async (symbol) => {
    try {
      const response = await fetch(`${BASE_URL}/stock/profile2?symbol=${symbol}&token=${API_KEY}`);
      return await response.json();
    } catch (error) {
      console.error('Error fetching company profile:', error);
      throw error;
    }
  },

  /**
   * Get current stock quote
   * @param {string} symbol - Stock symbol
   * @returns {Promise<Object>} - Current stock quote
   */
  getQuote: async (symbol) => {
    try {
      const response = await fetch(`${BASE_URL}/quote?symbol=${symbol}&token=${API_KEY}`);
      return await response.json();
    } catch (error) {
      console.error('Error fetching quote:', error);
      throw error;
    }
  },

  /**
   * Get historical stock data
   * @param {string} symbol - Stock symbol
   * @param {string} resolution - Data resolution (1, 5, 15, 30, 60, D, W, M)
   * @param {number} from - UNIX timestamp (seconds)
   * @param {number} to - UNIX timestamp (seconds)
   * @returns {Promise<Object>} - Historical stock data
   */
  getHistoricalData: async (symbol, resolution, from, to) => {
    try {
      const response = await fetch(
        `${BASE_URL}/stock/candle?symbol=${symbol}&resolution=${resolution}&from=${from}&to=${to}&token=${API_KEY}`
      );
      const data = await response.json();
      
      if (data.s === 'no_data') {
        throw new Error('No data available for the selected time range');
      }
      
      return data;
    } catch (error) {
      console.error('Error fetching historical data:', error);
      throw error;
    }
  },

  /**
   * Transform Finnhub candle data into a format suitable for charts
   * @param {Object} candleData - Raw candle data from Finnhub API
   * @returns {Array} - Formatted data for charts
   */
  formatCandleData: (candleData) => {
    if (!candleData || !candleData.t || candleData.s === 'no_data') {
      return [];
    }

    return candleData.t.map((timestamp, index) => {
      return {
        timestamp: new Date(timestamp * 1000),
        date: new Date(timestamp * 1000).toLocaleDateString(),
        open: candleData.o[index],
        high: candleData.h[index],
        low: candleData.l[index],
        close: candleData.c[index],
        volume: candleData.v[index],
      };
    });
  },

  /**
   * Format data specifically for line charts
   * @param {Array} formattedData - Data formatted by formatCandleData
   * @returns {Array} - Data formatted for line charts
   */
  formatForLineChart: (formattedData) => {
    return formattedData.map(item => ({
      x: item.date,
      y: item.close,
    }));
  },

  /**
   * Format stock search results to a simplified structure
   * @param {Array} searchResults - Raw search results from Finnhub API
   * @returns {Array} - Simplified search results
   */
  formatSearchResults: (searchResults) => {
    return searchResults.map(item => ({
      symbol: item.symbol,
      description: item.description,
      type: item.type,
    }));
  },

  /**
   * Calculate time periods for historical data queries
   * @returns {Object} - Common time periods in Unix timestamps
   */
  getTimePeriods: () => {
    const now = Math.floor(Date.now() / 1000);
    const oneDay = 24 * 60 * 60;
    
    return {
      oneDay: {
        from: now - oneDay,
        to: now,
        resolution: '5',
        label: '1D'
      },
      oneWeek: {
        from: now - (7 * oneDay),
        to: now,
        resolution: '15',
        label: '1W'
      },
      oneMonth: {
        from: now - (30 * oneDay),
        to: now,
        resolution: '60',
        label: '1M'
      },
      threeMonths: {
        from: now - (90 * oneDay),
        to: now,
        resolution: 'D',
        label: '3M'
      },
      oneYear: {
        from: now - (365 * oneDay),
        to: now,
        resolution: 'D',
        label: '1Y'
      },
      fiveYears: {
        from: now - (5 * 365 * oneDay),
        to: now,
        resolution: 'W',
        label: '5Y'
      }
    };
  }
};

export default FinnhubService; 