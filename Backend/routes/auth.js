const express = require('express');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const User = require('../models/User');

const router = express.Router();

// Register User
router.post('/register', async (req, res) => {
  try {
    // Extract fields from the request body
    const { 
      name, 
      email, 
      password, 
      countryCode, 
      phoneNumber, 
      address, 
      dateOfBirth, 
      captchaVerified 
    } = req.body;
    
    // Log received data
    console.log('Registration data received:', {
      name,
      email,
      passwordProvided: !!password,
      countryCode,
      phoneNumber,
      address,
      dateOfBirth,
      captchaVerified
    });
    
    // Validate required fields
    if (!name || !email || !password) {
      return res.status(400).json({ 
        success: false, 
        message: 'Name, email and password are required',
        validationErrors: {
          name: !name ? 'Name is required' : null,
          email: !email ? 'Email is required' : null,
          password: !password ? 'Password is required' : null,
          address:!address ?'Aeddress is reqired':null,
          dateOfBirth:!dateOfBirth ? 'Date Of Birth is Required':null
        }
      });
    }

    // Validate password length
    if (password.length < 6) {
      return res.status(400).json({ 
        success: false, 
        message: 'Password must be at least 6 characters long',
        validationErrors: {
          password: 'Password must be at least 6 characters long'
        }
      });
    }
    
    // Validate email format
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      return res.status(400).json({
        success: false,
        message: 'Invalid email format',
        validationErrors: {
          email: 'Invalid email format'
        }
      });
    }
    
    // Check if user exists
    const existingUser = await User.findOne({ email: email.toLowerCase() });
    if (existingUser) {
      return res.status(400).json({ 
        success: false, 
        message: 'User already exists',
        validationErrors: {
          email: 'Email is already registered'
        }
      });
    }
    
    // Check CAPTCHA verification
    if (!captchaVerified) {
      return res.status(403).json({
        success: false,
        message: 'CAPTCHA verification required',
        requiresCaptcha: true
      });
    }
    
    // Hash password
    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(password, salt);
    
    // Create current timestamp
    const currentTime = new Date();
    
    // Format date of birth
    let formattedDateOfBirth = null;
    if (dateOfBirth) {
      formattedDateOfBirth = new Date(dateOfBirth);
      if (isNaN(formattedDateOfBirth.getTime())) {
        return res.status(400).json({
          success: false,
          message: 'Invalid date of birth format',
          validationErrors: {
            dateOfBirth: 'Invalid date format'
          }
        });
      }
    }
    
    // Create new user with all fields
    const user = new User({
      name: name.trim(),
      email: email.toLowerCase().trim(),
      password: hashedPassword,
      countryCode: (countryCode || '+1').trim(),
      phoneNumber: (phoneNumber || '').trim(),
      address: (address || '').trim(),
      dateOfBirth: formattedDateOfBirth,
      captchaVerified: true,
      createdAt: currentTime,
      lastLogin: currentTime
    });
    
    // Log user object before saving
    console.log('User object before save:', user.toObject());
    
    // Save user to database
    await user.save();
    console.log('User saved successfully');
    
    // Generate JWT token
    const token = jwt.sign(
      { userId: user._id },
      process.env.JWT_SECRET || 'your-jwt-secret',
      { expiresIn: '7d' }
    );
    
    // Return user data and token
    res.status(201).json({
      success: true,
      token,
      user: {
        id: user._id,
        name: user.name,
        email: user.email,
        countryCode: user.countryCode,
        phoneNumber: user.phoneNumber,
        address: user.address,
        dateOfBirth: user.dateOfBirth,
        createdAt: user.createdAt,
        lastLogin: user.lastLogin,
        captchaVerified: user.captchaVerified
      }
    });
  } catch (error) {
    console.error('Registration error:', error);
    res.status(500).json({ 
      success: false, 
      message: 'Server error during registration',
      error: error.message
    });
  }
});

// Login User
router.post('/login', async (req, res) => {
  try {
    const { email, password } = req.body;
    
    // Input validation
    if (!email || !password) {
      return res.status(400).json({ success: false, message: 'All fields are required' });
    }
    
    // Check if user exists
    const user = await User.findOne({ email });
    if (!user) {
      return res.status(400).json({ success: false, message: 'Invalid credentials' });
    }
    
    // Verify password
    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) {
      return res.status(400).json({ success: false, message: 'Invalid credentials' });
    }
    
    // Check if CAPTCHA verification is required
    if (!user.captchaVerified) {
      return res.status(403).json({
        success: false,
        message: 'CAPTCHA verification required',
        requiresCaptcha: true,
      });
    }
    
    // Update last login time
    user.lastLogin = new Date();
    await user.save();
    
    // Get the updated user
    const updatedUser = await User.findById(user._id);
    if (!updatedUser) {
      throw new Error('Failed to retrieve updated user');
    }
    
    const userData = updatedUser.toObject();
    console.log('User data after login:', userData);
    
    // Generate JWT
    const token = jwt.sign(
      { userId: userData._id },
      process.env.JWT_SECRET || 'your-jwt-secret',
      { expiresIn: '7d' }
    );
    
    // Return ALL user data in response with consistent format
    res.json({
      success: true,
      token,
      user: {
        id: userData._id,
        name: userData.name || '',
        email: userData.email || '',
        countryCode: userData.countryCode || '+1',
        phoneNumber: userData.phoneNumber || '',
        address: userData.address || '',
        dateOfBirth: userData.dateOfBirth || '',
        createdAt: userData.createdAt || '',
        lastLogin: userData.lastLogin || '',
        captchaVerified: userData.captchaVerified || false
      }
    });
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ success: false, message: 'Server error' });
  }
});

module.exports = router; 