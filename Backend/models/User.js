const mongoose = require('mongoose');

// User Schema
const userSchema = new mongoose.Schema({
  email: { 
    type: String, 
    required: true, 
    unique: true,
    trim: true,
    lowercase: true
  },
  password: { 
    type: String, 
    required: true 
  },
  name: { 
    type: String, 
    required: true,
    trim: true
  },
  countryCode: { 
    type: String, 
    default: '+1',
    trim: true
  },
  phoneNumber: { 
    type: String, 
    default: '',
    trim: true
  },
  address: { 
    type: String, 
    default: '',
    trim: true
  },
  profilePicture: {
    type: String,
    default: '',
    trim: true
  },
  dateOfBirth: { 
    type: Date, 
    default: null,
    // Add validation to ensure date is in the past
    validate: {
      validator: function(value) {
        return !value || value <= new Date();
      },
      message: 'Date of birth must be in the past'
    }
  },
  captchaVerified: { 
    type: Boolean, 
    default: false 
  },
  createdAt: { 
    type: Date, 
    default: Date.now,
    immutable: true // Cannot be changed once set
  },
  lastLogin: { 
    type: Date, 
    default: null 
  }
});

// Add a pre-save hook to log the document being saved
userSchema.pre('save', function(next) {
  console.log('PRE-SAVE HOOK - DOCUMENT TO BE SAVED:');
  console.log(JSON.stringify(this.toObject(), null, 2));
  
  // Ensure dates are properly formatted
  if (this.dateOfBirth && typeof this.dateOfBirth === 'string') {
    this.dateOfBirth = new Date(this.dateOfBirth);
  }
  
  next();
});

const User = mongoose.model('User', userSchema);

module.exports = User; 