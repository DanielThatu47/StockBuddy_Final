import { API_URL } from '../config/constants';
import AsyncStorage from '@react-native-async-storage/async-storage';

const getAuthHeaders = async () => {
  const token = await AsyncStorage.getItem('userToken');
  return {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${token}`
  };
};

// Get all predictions for current user
export const getUserPredictions = async () => {
  try {
    const headers = await getAuthHeaders();
    const response = await fetch(`${API_URL}/api/predictions`, {
      method: 'GET',
      headers
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.message || 'Something went wrong');
    }
    
    return await response.json();
  } catch (error) {
    throw error;
  }
};

// Get a single prediction by ID
export const getPredictionById = async (id) => {
  try {
    const headers = await getAuthHeaders();
    const response = await fetch(`${API_URL}/api/predictions/${id}`, {
      method: 'GET',
      headers
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.message || 'Something went wrong');
    }
    
    return await response.json();
  } catch (error) {
    throw error;
  }
};

// Start a new prediction
export const startPrediction = async (symbol, daysAhead) => {
  try {
    const headers = await getAuthHeaders();
    const response = await fetch(`${API_URL}/api/predictions`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ symbol, daysAhead })
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.message || 'Failed to start prediction');
    }
    
    if (!data.taskId) {
      throw new Error('No task ID returned from server');
    }
    
    return data;
  } catch (error) {
    console.error('Prediction service error:', error);
    throw error;
  }
};

// Get prediction status
export const getPredictionStatus = async (taskId) => {
  try {
    const headers = await getAuthHeaders();
    const response = await fetch(`${API_URL}/api/predictions/status/${taskId}`, {
      method: 'GET',
      headers
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.message || 'Something went wrong');
    }
    
    return await response.json();
  } catch (error) {
    throw error;
  }
};

// Stop a prediction
export const stopPrediction = async (taskId) => {
  try {
    const headers = await getAuthHeaders();
    const response = await fetch(`${API_URL}/api/predictions/stop/${taskId}`, {
      method: 'POST',
      headers
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.message || 'Something went wrong');
    }
    
    return await response.json();
  } catch (error) {
    throw error;
  }
};

// Delete a single prediction
export const deletePrediction = async (id) => {
  try {
    const headers = await getAuthHeaders();
    const response = await fetch(`${API_URL}/api/predictions/${id}`, {
      method: 'DELETE',
      headers
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.message || 'Failed to delete prediction');
    }
    
    return await response.json();
  } catch (error) {
    throw error;
  }
};

// Delete multiple predictions
export const deleteMultiplePredictions = async (ids) => {
  try {
    const headers = await getAuthHeaders();
    const response = await fetch(`${API_URL}/api/predictions/delete-multiple`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ ids })
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.message || 'Failed to delete predictions');
    }
    
    return await response.json();
  } catch (error) {
    throw error;
  }
}; 