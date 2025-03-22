const express = require('express');
const cors = require('cors');
const session = require('express-session');
require('dotenv').config();

// Import routes
const authRoutes = require('./routes/auth');
const captchaRoutes = require('./routes/captcha');
const profileRoutes = require('./routes/profile');
const predictionRoutes = require('./routes/predictions');

// Import database connection
const connectDB = require('./config/db');

const app = express();
const PORT = process.env.PORT || 5000;

// Connect to MongoDB
connectDB();

// Middleware
app.use(cors({
  origin: '*', // Allow all origins
  credentials: true, // Allow credentials
  methods: ['GET', 'POST', 'PUT', 'DELETE'],
  allowedHeaders: ['Content-Type', 'Authorization']
}));

// Increase the limit for JSON payloads
app.use(express.json({ limit: '50mb' }));

// Increase the limit for URL-encoded payloads and ensure extended is true for nested objects
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Production middleware setup - minimal logging
if (process.env.NODE_ENV === 'production') {
  // In production, only log errors
  app.use((req, res, next) => {
    next();
  });
} else {
  // In development, log detailed request info
  app.use((req, res, next) => {
    const sanitizedBody = req.headers['content-type']?.includes('multipart/form-data')
      ? '[multipart form data]' 
      : req.body;

    console.log('Request received:', {
      method: req.method,
      url: req.url,
      headers: req.headers,
      body: sanitizedBody,
    });
    next();
  });
}

app.use(session({
  secret: process.env.SESSION_SECRET || 'your-session-secret',
  resave: false,
  saveUninitialized: true,
  cookie: { 
    secure: process.env.NODE_ENV === 'production',
    httpOnly: true,
    sameSite: 'none'
  }
}));

// Routes
app.use('/api', authRoutes);
app.use('/api', captchaRoutes);
app.use('/api/profile', profileRoutes);
app.use('/api/predictions', predictionRoutes);

// Basic route for testing
app.get('/', (req, res) => {
  res.send('API is running');
});

// Add environment variables for Cloudinary if needed
// You should also create a .env file with the following variables:
// CLOUDINARY_CLOUD_NAME=your_cloud_name
// CLOUDINARY_API_KEY=your_api_key
// CLOUDINARY_API_SECRET=your_api_secret

// Start server
app.listen(PORT, '0.0.0.0', () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`Server accessible at http://localhost:${PORT}`);
  console.log(`For Android emulator, use http://10.0.2.2:${PORT}`);
  console.log(`For devices on the same network, use http://<your-ip-address>:${PORT}`);
});