const express = require('express');
const svgCaptcha = require('svg-captcha');
const router = express.Router();

// Generate CAPTCHA
router.get('/captcha', (req, res) => {
  const captcha = svgCaptcha.create({
    size: 6,  // number of characters
    noise: 2, // number of noise lines
    color: true,
    width: 200,
    height: 100,
    fontSize: 72,
    charPreset: 'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjkmnpqrstuvwxyz23456789', // exclude confusing characters like 0, O, 1, I
    background: '#ffffff'
  });
  
  // Store captcha text in session
  req.session.captcha = captcha.text;
  console.log('Generated CAPTCHA text:', captcha.text);
  
  // Generate a simpler version for React Native
  const captchaText = captcha.text;
  
  // Response with both text representation and original SVG
  res.json({ 
    // Use text as primary method
    captchaText: captchaText,
    // Display text with spaces to make it more readable
    textRepresentation: `Verification Code: "${captchaText.split('').join(' ')}"`,
    timestamp: new Date().getTime()
  });
});

// Verify CAPTCHA
router.post('/verify-captcha', (req, res) => {
  const { userInput } = req.body;
  
  if (!req.session.captcha) {
    return res.status(400).json({ success: false, message: 'CAPTCHA session expired' });
  }
  
  // Case-sensitive comparison (exact match required)
  const isValid = userInput === req.session.captcha;
  
  // Clear the captcha from session after verification
  const originalCaptcha = req.session.captcha;
  delete req.session.captcha;
  
  if (isValid) {
    res.json({ success: true });
  } else {
    // Provide more detailed error message
    res.json({ 
      success: false, 
      message: 'Invalid CAPTCHA. Please ensure you enter the exact characters shown (case-sensitive).',
      expectedValue: originalCaptcha, // For debugging, remove in production
      userValue: userInput // For debugging, remove in production
    });
  }
});

module.exports = router; 