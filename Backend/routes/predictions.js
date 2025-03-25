const express = require('express');
const router = express.Router();
const Prediction = require('../models/Prediction');
const auth = require('../middleware/auth');
const fetch = require('node-fetch');
require('dotenv').config();

// Model API base URL
const MODEL_API_URL = process.env.MODEL_API_URL || 'http://localhost:5001';

// Get all predictions for the current user
router.get('/', auth, async (req, res) => {
  try {
    const predictions = await Prediction.find({ userId: req.userId })
      .sort({ createdAt: -1 });
    
    res.json(predictions);
  } catch (err) {
    console.error('Error fetching predictions:', err.message);
    res.status(500).json({ message: 'Server error' });
  }
});

// Get prediction by ID
router.get('/:id', auth, async (req, res) => {
  try {
    const prediction = await Prediction.findById(req.params.id);
    
    if (!prediction) {
      return res.status(404).json({ message: 'Prediction not found' });
    }
    
    // Check if the prediction belongs to the user
    if (prediction.userId.toString() !== req.userId) {
      return res.status(403).json({ message: 'Not authorized' });
    }
    
    res.json(prediction);
  } catch (err) {
    console.error('Error fetching prediction:', err.message);
    res.status(500).json({ message: 'Server error' });
  }
});

// Start a new prediction
router.post('/', auth, async (req, res) => {
  const { symbol, daysAhead } = req.body;
  
  if (!symbol) {
    return res.status(400).json({ message: 'Symbol is required' });
  }
  
  // Default to 3 days ahead for faster predictions
  const predictionDays = daysAhead || 3;
  
  try {
    // First check if there's any pending/running prediction for this user and symbol
    const existingPrediction = await Prediction.findOne({
      userId: req.userId,
      symbol: symbol.toUpperCase(),
      status: { $in: ['pending', 'running'] }
    });
    
    if (existingPrediction) {
      return res.status(400).json({ 
        message: 'You already have a pending prediction for this symbol. Please wait for it to complete or stop it first.' 
      });
    }
    
    // Call the model API to start prediction
    const response = await fetch(`${MODEL_API_URL}/api/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        userId: req.userId,
        symbol,
        daysAhead: predictionDays
      }),
      timeout: 10000 // Add 10 second timeout
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      return res.status(response.status).json(data);
    }
    
    // Ensure we have a valid taskId
    if (!data.taskId) {
      return res.status(500).json({ message: 'No taskId returned from prediction service' });
    }
    
    // Create a new prediction record
    const newPrediction = new Prediction({
      userId: req.userId,
      symbol: symbol.toUpperCase(),
      daysAhead: predictionDays,
      status: 'pending',
      taskId: data.taskId,
      predictions: []
    });
    
    await newPrediction.save();
    
    res.json(newPrediction);
  } catch (err) {
    console.error('Error starting prediction:', err.message);
    
    // Check for duplicate key error
    if (err.name === 'MongoServerError' && err.code === 11000) {
      // Retry with a new prediction_id
      try {
        const { v4: uuidv4 } = require('uuid');
        
        // Call the model API to start prediction
        const response = await fetch(`${MODEL_API_URL}/api/predict`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            userId: req.userId,
            symbol,
            daysAhead: predictionDays
          }),
        });
        
        const data = await response.json();
        
        if (!response.ok) {
          return res.status(response.status).json(data);
        }
        
        // Ensure we have a valid taskId
        if (!data.taskId) {
          return res.status(500).json({ message: 'No taskId returned from prediction service' });
        }
        
        // Create a new prediction record with explicitly set prediction_id
        const newPrediction = new Prediction({
          userId: req.userId,
          symbol: symbol.toUpperCase(),
          daysAhead: predictionDays,
          status: 'pending',
          taskId: data.taskId,
          predictions: [],
          prediction_id: uuidv4()
        });
        
        await newPrediction.save();
        
        return res.json(newPrediction);
      } catch (retryErr) {
        console.error('Error in retry:', retryErr.message);
        return res.status(400).json({ 
          message: 'A prediction with this ID already exists. Please try again in a moment.' 
        });
      }
    }
    
    res.status(500).json({ message: 'Server error' });
  }
});

// Get prediction status
router.get('/status/:taskId', auth, async (req, res) => {
  const { taskId } = req.params;
  
  try {
    // Find the prediction in our database
    const prediction = await Prediction.findOne({ taskId });
    
    if (!prediction) {
      return res.status(404).json({ message: 'Prediction not found' });
    }
    
    // Check if the prediction belongs to the user
    if (prediction.userId.toString() !== req.userId) {
      return res.status(403).json({ message: 'Not authorized' });
    }
    
    // Call the model API to get status
    const response = await fetch(`${MODEL_API_URL}/api/predict/status/${taskId}`);
    
    try {
      const data = await response.json();
      
      if (!response.ok) {
        return res.status(response.status).json(data);
      }
      
      // Update the prediction if completed
      if (data.status === 'completed' && prediction.status !== 'completed') {
        prediction.status = 'completed';
        if (data.result && data.result.predictions) {
          prediction.predictions = data.result.predictions;
          prediction.sentiment = data.result.sentiment;
          await prediction.save();
        } else {
          console.error('Completed prediction missing predictions data:', data);
          return res.status(500).json({ message: 'Invalid prediction data format' });
        }
      } else if (data.status === 'failed' && prediction.status !== 'failed') {
        prediction.status = 'failed';
        prediction.error = data.error || 'Unknown error occurred';
        await prediction.save();
      } else if (data.status !== prediction.status) {
        prediction.status = data.status;
        await prediction.save();
      }
      
      res.json({
        ...data,
        id: prediction._id
      });
    } catch (jsonError) {
      console.error('Error parsing prediction status JSON response:', jsonError.message);
      
      // Update prediction to error state if parsing fails
      prediction.status = 'failed';
      prediction.error = 'Error parsing prediction result';
      await prediction.save();
      
      return res.status(500).json({ 
        message: 'Invalid response from prediction service', 
        error: jsonError.message 
      });
    }
  } catch (err) {
    console.error('Error checking prediction status:', err.message);
    res.status(500).json({ message: 'Server error' });
  }
});

// Stop a prediction
router.post('/stop/:taskId', auth, async (req, res) => {
  const { taskId } = req.params;
  
  try {
    // Find the prediction in our database
    const prediction = await Prediction.findOne({ taskId });
    
    if (!prediction) {
      return res.status(404).json({ message: 'Prediction not found' });
    }
    
    // Check if the prediction belongs to the user
    if (prediction.userId.toString() !== req.userId) {
      return res.status(403).json({ message: 'Not authorized' });
    }
    
    // Call the model API to stop prediction
    const response = await fetch(`${MODEL_API_URL}/api/predict/stop/${taskId}`, {
      method: 'POST'
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      return res.status(response.status).json(data);
    }
    
    // Update the prediction status
    prediction.status = 'stopped';
    await prediction.save();
    
    res.json({
      ...data,
      id: prediction._id
    });
  } catch (err) {
    console.error('Error stopping prediction:', err.message);
    res.status(500).json({ message: 'Server error' });
  }
});

module.exports = router; 