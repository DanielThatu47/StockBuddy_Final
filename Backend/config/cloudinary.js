const cloudinary = require('cloudinary').v2;
const multer = require('multer');
const path = require('path');
const fs = require('fs');
require('dotenv').config();

// Configure Cloudinary with your account credentials
cloudinary.config({
  cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
  api_key: process.env.CLOUDINARY_API_KEY,
  api_secret: process.env.CLOUDINARY_API_SECRET,
  secure: true // Use HTTPS
});

// Create temp directory for uploads if it doesn't exist
const tempDir = path.join(__dirname, '../temp');
if (!fs.existsSync(tempDir)) {
  fs.mkdirSync(tempDir, { recursive: true });
}

// Configure multer disk storage instead of cloudinary storage
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, tempDir);
  },
  filename: function (req, file, cb) {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname));
  }
});

// Configure multer upload with error handling
const upload = multer({
  storage: storage,
  limits: {
    fileSize: 5 * 1024 * 1024 // 5MB max file size
  },
  fileFilter: (req, file, cb) => {
    // Accept only jpg, jpeg, and png
    if (file.mimetype === 'image/jpeg' || file.mimetype === 'image/jpg' || file.mimetype === 'image/png') {
      cb(null, true);
    } else {
      cb(new Error('Unsupported file type. Please upload only JPG or PNG images.'), false);
    }
  }
});

// Function to upload file to Cloudinary with optimizations
const uploadToCloudinary = async (filePath) => {
  try {
    // Upload the file to Cloudinary with performance optimizations
    const result = await cloudinary.uploader.upload(filePath, {
      folder: 'profile-pictures',
      use_filename: true,
      // Add transformations for better performance
      transformation: [
        { width: 500, height: 500, crop: 'limit' },
        { quality: 'auto:good', fetch_format: 'auto' }
      ],
      // Add eager transformations to generate optimized versions upfront
      eager: [
        // Small avatar version
        { width: 150, height: 150, crop: 'fill', quality: 'auto:good', fetch_format: 'auto' },
        // Medium avatar version
        { width: 300, height: 300, crop: 'fill', quality: 'auto:good', fetch_format: 'auto' }
      ],
      // Enable image analysis to remove extra image data
      colors: false,
      faces: false,
      exif: false,
      metadata: false,
      // Cache optimization
      resource_type: 'image',
      type: 'upload',
      overwrite: true,
      invalidate: true
    });

    // Delete the local file after upload
    fs.unlinkSync(filePath);

    // Return Cloudinary result
    return result;
  } catch (error) {
    // Delete the local file in case of error
    if (fs.existsSync(filePath)) {
      fs.unlinkSync(filePath);
    }
    throw error;
  }
};

// Function to delete image from Cloudinary
const deleteFromCloudinary = async (imageUrl) => {
  try {
    // Extract the public_id from the Cloudinary URL
    if (!imageUrl || typeof imageUrl !== 'string') {
      console.warn('Invalid image URL provided for deletion');
      return { result: 'not_deleted', reason: 'invalid_url' };
    }
    
    // Parse Cloudinary URL to get public ID
    const publicId = getPublicIdFromUrl(imageUrl);
    
    if (!publicId) {
      console.warn('Could not extract public ID from URL:', imageUrl);
      return { result: 'not_deleted', reason: 'invalid_public_id' };
    }
    
    console.log(`Deleting image with public ID: ${publicId}`);
    
    // Delete the image from Cloudinary
    const result = await cloudinary.uploader.destroy(publicId, {
      invalidate: true // Invalidate CDN cache for this image
    });
    
    console.log('Cloudinary delete result:', result);
    return result;
  } catch (error) {
    console.error('Error deleting from Cloudinary:', error);
    throw error;
  }
};

// Helper function to extract public_id from Cloudinary URL
const getPublicIdFromUrl = (url) => {
  try {
    if (!url) return null;
    
    // Extract the public ID from the URL
    // Format: https://res.cloudinary.com/{cloud_name}/image/upload/v{version}/{public_id}.{format}
    const regex = /\/(?:v\d+\/)?([^/\.]+)/;
    const match = url.match(regex);
    
    if (match && match[1]) {
      // Extract the folder and public ID
      const fullPath = match[1];
      
      // If the path contains 'profile-pictures/', include it in the public ID
      if (url.includes('profile-pictures/')) {
        return 'profile-pictures/' + fullPath.split('profile-pictures/').pop();
      }
      
      return fullPath;
    }
    
    return null;
  } catch (error) {
    console.error('Error extracting public ID:', error);
    return null;
  }
};

module.exports = {
  cloudinary,
  upload,
  uploadToCloudinary,
  deleteFromCloudinary
}; 