const mongoose = require('mongoose');
const { v4: uuidv4 } = require('uuid');

const predictionSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  symbol: {
    type: String,
    required: true,
    trim: true,
    uppercase: true
  },
  daysAhead: {
    type: Number,
    required: true,
    min: 1,
    max: 30
  },
  predictions: [{
    date: {
      type: String,
      required: true
    },
    price: {
      type: Number,
      required: true
    }
  }],
  sentiment: {
    totals: {
      positive: Number,
      negative: Number,
      neutral: Number
    },
    summary: String
  },
  createdAt: {
    type: Date,
    default: Date.now,
    immutable: true
  },
  status: {
    type: String,
    enum: ['pending', 'running', 'completed', 'failed', 'stopped'],
    default: 'pending'
  },
  taskId: {
    type: String,
    required: true,
    unique: true,
    trim: true,
    validate: {
      validator: function(v) {
        return v && v.length > 0;
      },
      message: 'TaskId cannot be empty'
    }
  },
  prediction_id: {
    type: String,
    unique: true,
    default: () => uuidv4()
  }
});

// Add index for faster queries
predictionSchema.index({ userId: 1, status: 1 });
predictionSchema.index({ userId: 1, symbol: 1, status: 1 });

// Pre-save hook to ensure taskId is present
predictionSchema.pre('save', function(next) {
  if (!this.taskId) {
    const err = new Error('TaskId is required');
    next(err);
  } else {
    next();
  }
});

const Prediction = mongoose.model('Prediction', predictionSchema);

module.exports = Prediction; 