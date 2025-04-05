const express = require('express');
const router = express.Router();
const User = require('../models/User');
const auth = require('../middleware/auth');
const { upload, uploadToCloudinary, deleteFromCloudinary } = require('../config/cloudinary');

// GET: Fetch user profile
router.get('/', auth, async (req, res) => {
  try {
    const user = await User.findById(req.userId).select('-password');
    if (!user) {
      return res.status(404).json({ success: false, message: 'User not found' });
    }

    const userData = user.toObject();
    return res.status(200).json({
      success: true,
      user: {
        id: userData._id,
        name: userData.name || '',
        email: userData.email || '',
        countryCode: userData.countryCode || '+1',
        phoneNumber: userData.phoneNumber || '',
        address: userData.address || '',
        profilePicture: userData.profilePicture || '',
        dateOfBirth: userData.dateOfBirth || null,
        createdAt: userData.createdAt || null,
        lastLogin: userData.lastLogin || null,
        captchaVerified: userData.captchaVerified || false
      }
    });
  } catch (error) {
    return res.status(500).json({ success: false, message: 'Server error during profile fetch' });
  }
});

// PUT: Update user profile (excluding image)
router.put('/', auth, async (req, res) => {
  try {
    const { name, email, countryCode, phoneNumber, address, profilePicture } = req.body;
    const user = await User.findById(req.userId);
    if (!user) return res.status(404).json({ success: false, message: 'User not found' });

    if (email && email !== user.email) {
      const existingUser = await User.findOne({ email });
      if (existingUser) {
        return res.status(400).json({ success: false, message: 'Email already in use' });
      }
      user.email = email;
    }

    user.name = name || user.name;
    user.countryCode = countryCode || user.countryCode;
    user.phoneNumber = phoneNumber !== undefined ? phoneNumber : user.phoneNumber;
    user.address = address !== undefined ? address : user.address;
    user.profilePicture = profilePicture !== undefined ? profilePicture : user.profilePicture;

    await user.save();

    const userData = user.toObject();
    return res.json({
      success: true,
      user: {
        id: userData._id,
        name: userData.name || '',
        email: userData.email || '',
        countryCode: userData.countryCode || '+1',
        phoneNumber: userData.phoneNumber || '',
        address: userData.address || '',
        profilePicture: userData.profilePicture || '',
        dateOfBirth: userData.dateOfBirth || null,
        createdAt: userData.createdAt || null,
        lastLogin: userData.lastLogin || null,
        captchaVerified: userData.captchaVerified || false
      },
    });
  } catch (error) {
    return res.status(500).json({ success: false, message: 'Server error' });
  }
});

// POST: Upload new profile picture
router.post('/upload-picture', auth, (req, res) => {
  upload.single('image')(req, res, async (err) => {
    if (err) {
      console.error('Upload error:', err.message);
      return res.status(400).json({ success: false, message: `Upload error: ${err.message}` });
    }

    if (!req.file) {
      return res.status(400).json({ success: false, message: 'No file uploaded' });
    }

    try {
      const user = await User.findById(req.userId);
      if (!user) return res.status(404).json({ success: false, message: 'User not found' });

      if (user.profilePicture) {
        try {
          await deleteFromCloudinary(user.profilePicture);
        } catch (delErr) {
          console.warn('Old profile picture deletion failed:', delErr.message);
        }
      }

      const result = await uploadToCloudinary(req.file.path);
      user.profilePicture = result.secure_url;
      await user.save();

      const userData = user.toObject();
      return res.status(200).json({
        success: true,
        profilePicture: result.secure_url,
        user: {
          id: userData._id,
          name: userData.name || '',
          email: userData.email || '',
          countryCode: userData.countryCode || '+1',
          phoneNumber: userData.phoneNumber || '',
          address: userData.address || '',
          profilePicture: userData.profilePicture || '',
          dateOfBirth: userData.dateOfBirth || null,
          createdAt: userData.createdAt || null,
          lastLogin: userData.lastLogin || null,
          captchaVerified: userData.captchaVerified || false
        }
      });
    } catch (error) {
      console.error('Picture processing failed:', error.message);
      return res.status(500).json({ success: false, message: 'Server error during image processing' });
    }
  });
});

// DELETE: Remove profile picture
router.delete('/profile-picture', auth, async (req, res) => {
  try {
    const user = await User.findById(req.userId);
    if (!user) return res.status(404).json({ success: false, message: 'User not found' });

    if (user.profilePicture) {
      try {
        await deleteFromCloudinary(user.profilePicture);
      } catch (err) {
        console.warn('Error deleting from Cloudinary:', err.message);
      }
      user.profilePicture = '';
      await user.save();
    }

    const userData = user.toObject();
    return res.status(200).json({
      success: true,
      message: 'Profile picture removed',
      user: {
        id: userData._id,
        name: userData.name || '',
        email: userData.email || '',
        countryCode: userData.countryCode || '+1',
        phoneNumber: userData.phoneNumber || '',
        address: userData.address || '',
        profilePicture: userData.profilePicture || '',
        dateOfBirth: userData.dateOfBirth || null,
        createdAt: userData.createdAt || null,
        lastLogin: userData.lastLogin || null,
        captchaVerified: userData.captchaVerified || false
      }
    });
  } catch (error) {
    console.error('Error during profile picture removal:', error.message);
    return res.status(500).json({ success: false, message: 'Server error during profile picture removal' });
  }
});

// DELETE: User account
router.delete('/', auth, async (req, res) => {
  try {
    const user = await User.findById(req.userId);
    if (!user) return res.status(404).json({ success: false, message: 'User not found' });

    await User.findByIdAndDelete(req.userId);
    return res.status(200).json({ success: true, message: 'Account successfully deleted' });
  } catch (error) {
    return res.status(500).json({ success: false, message: 'Server error during account deletion' });
  }
});

module.exports = router;