const cloudinary = require('cloudinary').v2;
const multer = require('multer');
require('dotenv').config();

// Configure Cloudinary
cloudinary.config({
  cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
  api_key: process.env.CLOUDINARY_API_KEY,
  api_secret: process.env.CLOUDINARY_API_SECRET,
  secure: true
});

// Use memory storage instead of writing to disk
const storage = multer.memoryStorage();

const upload = multer({
  storage: storage,
  limits: {
    fileSize: 5 * 1024 * 1024 // 5MB max
  },
  fileFilter: (req, file, cb) => {
    if (['image/jpeg', 'image/jpg', 'image/png'].includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error('Unsupported file type. Please upload only JPG or PNG images.'), false);
    }
  }
});

// Upload buffer to Cloudinary using stream
const uploadToCloudinary = async (fileBuffer) => {
  return new Promise((resolve, reject) => {
    const stream = cloudinary.uploader.upload_stream({
      folder: 'profile-pictures',
      resource_type: 'image',
      use_filename: true,
      transformation: [
        { width: 500, height: 500, crop: 'limit' },
        { quality: 'auto:good', fetch_format: 'auto' }
      ],
      eager: [
        { width: 150, height: 150, crop: 'fill', quality: 'auto:good', fetch_format: 'auto' },
        { width: 300, height: 300, crop: 'fill', quality: 'auto:good', fetch_format: 'auto' }
      ],
      invalidate: true
    }, (error, result) => {
      if (error) return reject(error);
      resolve(result);
    });

    stream.end(fileBuffer);
  });
};

// Delete image from Cloudinary
const deleteFromCloudinary = async (imageUrl) => {
  try {
    const publicId = getPublicIdFromUrl(imageUrl);
    if (!publicId) return { result: 'not_deleted', reason: 'invalid_public_id' };

    const result = await cloudinary.uploader.destroy(publicId, {
      invalidate: true
    });

    return result;
  } catch (error) {
    console.error('Error deleting from Cloudinary:', error);
    throw error;
  }
};

// Extract public ID from Cloudinary URL
const getPublicIdFromUrl = (url) => {
  try {
    if (!url) return null;

    const matches = url.match(/\/(?:v\d+\/)?([^\/]+)(?:\.[a-z]+)?$/i);
    const publicId = matches && matches[1];

    if (url.includes('profile-pictures/')) {
      return 'profile-pictures/' + publicId;
    }

    return publicId;
  } catch (error) {
    return null;
  }
};

module.exports = {
  cloudinary,
  upload,
  uploadToCloudinary,
  deleteFromCloudinary
};