const express = require('express');
const router = express.Router();
const User = require('../models/User');
const auth = require('../middleware/auth');
const { upload, uploadToCloudinary, deleteFromCloudinary } = require('../config/cloudinary');

// Get user profile
router.get('/', auth, async (req, res) => {
  try {
    const user = await User.findById(req.userId).select('-password');
    
    if (!user) {
      return res.status(404).json({ success: false, message: 'User not found' });
    }
    
    // Get the raw user data
    const userData = user.toObject();
    
    // Format and return the user data
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

// Update user profile
router.put('/', auth, async (req, res) => {
  try {
    const { name, email, countryCode, phoneNumber, address, profilePicture } = req.body;
    const user = await User.findById(req.userId);
    
    if (!user) {
      return res.status(404).json({ success: false, message: 'User not found' });
    }
    
    if (email && email !== user.email) {
      const existingUser = await User.findOne({ email });
      if (existingUser) {
        return res.status(400).json({ success: false, message: 'Email already in use' });
      }
      user.email = email;
    }
    
    if (name) {
      user.name = name;
    }
    
    if (countryCode) {
      user.countryCode = countryCode;
    }
    
    if (phoneNumber !== undefined) {
      user.phoneNumber = phoneNumber;
    }
    
    if (address !== undefined) {
      user.address = address;
    }
    
    if (profilePicture !== undefined) {
      user.profilePicture = profilePicture;
    }
    
    await user.save();
    
    const userData = user.toObject();
    
    res.json({
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
    res.status(500).json({ success: false, message: 'Server error' });
  }
});

// Upload profile picture
router.post('/upload-picture', auth, async (req, res) => {
  try {
    console.log('Upload picture request received');
    
    // Use multer to handle the file upload
    upload.single('image')(req, res, async (err) => {
      if (err) {
        console.error('File upload error:', err.message);
        return res.status(400).json({ success: false, message: `File upload error: ${err.message}` });
      }
      
      // Check if file was uploaded
      if (!req.file) {
        console.error('No file uploaded');
        return res.status(400).json({ success: false, message: 'No file uploaded' });
      }
      
      try {
        console.log('File received, uploading to Cloudinary:', req.file.path);
        
        // If user already has a profile picture, delete it first
        const user = await User.findById(req.userId);
        
        if (!user) {
          console.error('User not found');
          return res.status(404).json({ success: false, message: 'User not found' });
        }
        
        // Delete old profile picture if it exists
        if (user.profilePicture) {
          try {
            console.log('Deleting old profile picture:', user.profilePicture);
            await deleteFromCloudinary(user.profilePicture);
            console.log('Old profile picture deleted successfully');
          } catch (deleteError) {
            console.error('Error deleting old profile picture:', deleteError);
            // Continue even if deletion fails
          }
        }
        
        // Upload the file to Cloudinary
        const cloudinaryResult = await uploadToCloudinary(req.file.path);
        console.log('Cloudinary upload successful:', cloudinaryResult.secure_url);
        
        // Update profile picture URL in the database
        user.profilePicture = cloudinaryResult.secure_url;
        await user.save();
        
        const userData = user.toObject();
        
        return res.status(200).json({
          success: true,
          profilePicture: cloudinaryResult.secure_url,
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
      } catch (innerError) {
        console.error('Server error during picture processing:', innerError);
        return res.status(500).json({ success: false, message: 'Server error during picture processing' });
      }
    });
  } catch (error) {
    console.error('Server error during picture upload:', error);
    return res.status(500).json({ success: false, message: 'Server error during picture upload' });
  }
});

// Delete profile picture
router.delete('/profile-picture', auth, async (req, res) => {
  try {
    // Get the user
    const user = await User.findById(req.userId);
    
    if (!user) {
      return res.status(404).json({ success: false, message: 'User not found' });
    }
    
    // If the user has a profile picture, delete it from Cloudinary
    if (user.profilePicture) {
      try {
        console.log('Attempting to delete profile picture from Cloudinary:', user.profilePicture);
        
        // Delete image from Cloudinary using the full URL
        // The deleteFromCloudinary function will extract the public_id
        await deleteFromCloudinary(user.profilePicture);
        console.log('Successfully deleted image from Cloudinary');
      } catch (cloudinaryError) {
        console.error('Error deleting from Cloudinary:', cloudinaryError);
        // Continue execution even if Cloudinary deletion fails
      }
      
      // Clear the profile picture field
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
    console.error('Server error during picture deletion:', error);
    return res.status(500).json({ success: false, message: 'Server error during picture deletion' });
  }
});

// Delete user account
router.delete('/', auth, async (req, res) => {
  try {
    // Find the user by ID
    const user = await User.findById(req.userId);
    
    if (!user) {
      return res.status(404).json({ success: false, message: 'User not found' });
    }
    
    // Delete the user
    await User.findByIdAndDelete(req.userId);
    
    // Return success response
    return res.status(200).json({ success: true, message: 'Account successfully deleted' });
  } catch (error) {
    return res.status(500).json({ success: false, message: 'Server error during account deletion' });
  }
});

module.exports = router; 