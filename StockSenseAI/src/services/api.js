import AsyncStorage from '@react-native-async-storage/async-storage';

// Base URL for all API requests 
// Using 10.0.2.2 which is the special IP for the Android emulator to access the localhost of the host machine
const BASE_URL = 'http://192.168.6.97:5000/api';

/**
 * Generic API request function
 * @param {string} endpoint - API endpoint to call
 * @param {string} method - HTTP method (GET, POST, PUT, DELETE)
 * @param {object} body - Request body for POST/PUT requests
 * @param {boolean} requiresAuth - Whether this request requires authentication
 * @returns {Promise} - Promise with the API response
 */
export const apiRequest = async (endpoint, method = 'GET', body = null, requiresAuth = false) => {
  try {
    // Log request details
    console.log(`API Request: ${method} ${BASE_URL}${endpoint}`);
    if (body) console.log('Request body:', JSON.stringify(body));
    
    // Configure request headers
    const headers = {
      'Content-Type': 'application/json',
    };
    
    // Add authorization token if required
    if (requiresAuth) {
      const token = await AsyncStorage.getItem('userToken');
      if (!token) {
        throw new Error('Authentication required');
      }
      headers['Authorization'] = `Bearer ${token}`;
    }
    
    // Configure the request
    const config = {
      method,
      headers,
    };
    
    // Add request body if provided
    if (body) {
      config.body = JSON.stringify(body);
    }
    
    // Make the API call
    const response = await fetch(`${BASE_URL}${endpoint}`, config);
    
    // Log response status
    console.log(`Response status: ${response.status}`);
    
    // Parse the response
    const data = await response.json();
    console.log('Response data:', JSON.stringify(data));
    
    // Handle API errors
    if (!response.ok) {
      throw new Error(data.message || 'An error occurred');
    }
    
    return data;
  } catch (error) {
    console.error('API request failed:', error.message);
    throw error;
  }
};

// Authentication service
export const authService = {
  // Login user
  login: async (email, password) => {
    return apiRequest('/login', 'POST', { email, password });
  },
  
  // Register user
  register: async (userData) => {
    return apiRequest('/register', 'POST', { 
      ...userData,
      captchaVerified: true // Set CAPTCHA as verified by default for now
    });
  },
  
  // Get user profile
  getUserProfile: async () => {
    return apiRequest('/profile', 'GET', null, true);
  },
  
  // Update user profile
  updateProfile: async (profileData) => {
    return apiRequest('/profile', 'PUT', profileData, true);
  }
};

export default {
  apiRequest,
  authService
}; 