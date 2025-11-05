// server.js - Node.js Express Backend for Activity Annotations
// Videos stored in memory only, annotations saved to MongoDB

const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const session = require('express-session');
const MongoStore = require('connect-mongo');
const multer = require('multer');
const csv = require('csv-parser');
const { createObjectCsvWriter } = require('csv-writer');
const fs = require('fs');
const path = require('path');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors({
  origin: process.env.CLIENT_URL || 'http://localhost:8080',
  credentials: true
}));
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Session configuration
app.use(session({
  secret: process.env.SESSION_SECRET || 'your-secret-key-change-this',
  resave: false,
  saveUninitialized: false,
  store: MongoStore.create({
    mongoUrl: process.env.MONGODB_URI || 'mongodb://localhost:27017/activity_annotations'
  }),
  cookie: {
    maxAge: 1000 * 60 * 60 * 24 * 7 // 1 week
  }
}));

// MongoDB Connection
mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/activity_annotations', {
  useNewUrlParser: true,
  useUnifiedTopology: true
}).then(() => {
  console.log('✓ Connected to MongoDB');
}).catch(err => {
  console.error('✗ MongoDB connection error:', err);
});

const db = mongoose.connection;

// Schemas
const UserSchema = new mongoose.Schema({
  name: { type: String, required: true, unique: true },
  createdAt: { type: Date, default: Date.now }
});

const AnnotationSchema = new mongoose.Schema({
  annotatorName: { type: String, required: true, index: true },
  videoFilename: { type: String, required: true, index: true },
  description: String,
  childActivities: [String],
  childActivityConfidence: { type: String, required: true },
  otherPersonPresent: { type: String, required: true },
  otherPersonType: [String],
  otherPersonActivities: [String],
  otherPersonConfidence: String,
  sameSpace: String,
  sameActivity: String,
  childPosture: [String],
  childPostureConfidence: { type: String, required: true },
  locations: [String],
  locationConfidence: { type: String, required: true },
  descriptionRating: { type: String, required: true },
  updatedAt: { type: Date, default: Date.now }
});

AnnotationSchema.index({ annotatorName: 1, videoFilename: 1 });

const User = mongoose.model('User', UserSchema);
const Annotation = mongoose.model('Annotation', AnnotationSchema);

const useBasicAuth = process.env.USE_BASIC_AUTH === 'true';
// Basic Auth Middleware
const basicAuth = (req, res, next) => {
  if (!useBasicAuth) {
    return next();
  }
  const authHeader = req.headers.authorization;
  
  if (!authHeader) {
    res.setHeader('WWW-Authenticate', 'Basic realm="Activity Annotations"');
    return res.status(401).json({ error: 'Authentication required' });
  }

  const auth = Buffer.from(authHeader.split(' ')[1], 'base64').toString().split(':');
  const username = auth[0];
  const password = auth[1];

  if (username === process.env.APP_USERNAME && password === process.env.APP_PASSWORD) {
    next();
  } else {
    res.setHeader('WWW-Authenticate', 'Basic realm="Activity Annotations"');
    return res.status(401).json({ error: 'Invalid credentials' });
  }
};

// Create uploads directory for temporary CSV files only
const uploadsDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir);
}

// File upload configuration - CSV only (videos not stored)
const csvUpload = multer({ dest: uploadsDir });

// Serve static files
app.use('/experiment', express.static('public'));

// Routes

// Health check
app.get('/api/health', basicAuth, (req, res) => {
  res.json({ status: 'ok', timestamp: new Date(), mongodb: mongoose.connection.readyState === 1 });
});

// Login/Create User
app.post('/api/login', basicAuth, async (req, res) => {
  try {
    const { name } = req.body;
    
    if (!name || !name.trim()) {
      return res.status(400).json({ error: 'Name is required' });
    }

    let user = await User.findOne({ name: name.trim() });
    
    if (!user) {
      user = new User({ name: name.trim() });
      await user.save();
    }

    req.session.annotatorName = user.name;
    
    res.json({ 
      success: true, 
      user: { name: user.name, createdAt: user.createdAt }
    });
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get current user
app.get('/api/user', basicAuth, (req, res) => {
  if (req.session.annotatorName) {
    res.json({ name: req.session.annotatorName });
  } else {
    res.status(401).json({ error: 'Not logged in' });
  }
});

// Logout
app.post('/api/logout', basicAuth, (req, res) => {
  req.session.destroy();
  res.json({ success: true });
});

// Parse CSV and return video list (videos handled client-side)
app.post('/api/upload-csv', basicAuth, csvUpload.single('csvFile'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    const videos = [];
    const filePath = req.file.path;

    fs.createReadStream(filePath)
      .pipe(csv())
      .on('data', (row) => {
        const videoFilename = row.video_filename || row.videoFilename;
        videos.push({
          videoFilename: videoFilename,
          description: row.description || '',
          order: parseInt(row.order) || videos.length + 1
        });
      })
      .on('end', async () => {
        // Clean up uploaded CSV file
        fs.unlinkSync(filePath);

        // Sort by order
        videos.sort((a, b) => a.order - b.order);

        // Get existing annotations for this user
        const annotatorName = req.session.annotatorName;
        const videoFilenames = videos.map(v => v.videoFilename);
        
        const existingAnnotations = await Annotation.find({
          annotatorName,
          videoFilename: { $in: videoFilenames }
        });

        // Create a map of existing annotations
        const annotationMap = {};
        existingAnnotations.forEach(ann => {
          annotationMap[ann.videoFilename] = ann;
        });

        // Add existing annotations to video data
        videos.forEach(video => {
          if (annotationMap[video.videoFilename]) {
            video.existingAnnotation = annotationMap[video.videoFilename];
          }
        });
        
        res.json({ 
          success: true, 
          videos,
          totalCount: videos.length
        });
      })
      .on('error', (error) => {
        fs.unlinkSync(filePath);
        res.status(500).json({ error: error.message });
      });
  } catch (error) {
    console.error('CSV upload error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get existing annotations for a video
app.get('/api/annotations/:videoFilename', basicAuth, async (req, res) => {
  try {
    const annotatorName = req.session.annotatorName;
    if (!annotatorName) {
      return res.status(401).json({ error: 'Not logged in' });
    }

    const annotation = await Annotation.findOne({
      annotatorName,
      videoFilename: req.params.videoFilename
    });

    if (annotation) {
      res.json({ success: true, annotation });
    } else {
      res.json({ success: true, annotation: null });
    }
  } catch (error) {
    console.error('Get annotation error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Save annotation
app.post('/api/annotations', basicAuth, async (req, res) => {
  try {
    const annotatorName = req.session.annotatorName;
    if (!annotatorName) {
      return res.status(401).json({ error: 'Not logged in' });
    }

    const annotationData = {
      annotatorName,
      videoFilename: req.body.videoFilename,
      description: req.body.description || '',
      childActivities: req.body.childActivities || [],
      childActivityConfidence: req.body.childActivityConfidence,
      otherPersonPresent: req.body.otherPersonPresent,
      childPosture: req.body.childPosture,
      childPostureConfidence: req.body.childPostureConfidence,
      locations: req.body.locations || [],
      locationConfidence: req.body.locationConfidence,
      descriptionRating: req.body.descriptionRating,
      updatedAt: new Date()
    };

    // Add other person fields if present
    if (req.body.otherPersonPresent === 'yes') {
      annotationData.otherPersonType = req.body.otherPersonType;
      annotationData.otherPersonActivities = req.body.otherPersonActivities || [];
      annotationData.otherPersonConfidence = req.body.otherPersonConfidence;
      annotationData.sameSpace = req.body.sameSpace;
      annotationData.sameActivity = req.body.sameActivity;
    }

    const result = await Annotation.findOneAndUpdate(
      { annotatorName, videoFilename: req.body.videoFilename },
      annotationData,
      { upsert: true, new: true }
    );

    res.json({ success: true, annotation: result });
  } catch (error) {
    console.error('Save annotation error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get all annotations for current user
app.get('/api/annotations', basicAuth, async (req, res) => {
  try {
    const annotatorName = req.session.annotatorName;
    if (!annotatorName) {
      return res.status(401).json({ error: 'Not logged in' });
    }

    const annotations = await Annotation.find({ annotatorName })
      .sort({ updatedAt: -1 });

    res.json({ success: true, annotations, count: annotations.length });
  } catch (error) {
    console.error('Get annotations error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Export annotations to CSV
app.get('/api/export', async (req, res) => {
  try {
    const annotatorName = req.session.annotatorName;
    if (!annotatorName) {
      return res.status(401).json({ error: 'Not logged in' });
    }

    const annotations = await Annotation.find({ annotatorName })
      .sort({ updatedAt: -1 })
      .lean();

    if (annotations.length === 0) {
      return res.status(400).json({ error: 'No annotations to export' });
    }

    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `annotations_${annotatorName}_${timestamp}.csv`;
    const filepath = path.join(__dirname, 'exports', filename);

    // Ensure exports directory exists
    if (!fs.existsSync(path.join(__dirname, 'exports'))) {
      fs.mkdirSync(path.join(__dirname, 'exports'));
    }

    // Flatten arrays to comma-separated strings
    const flattenedAnnotations = annotations.map(ann => ({
      ...ann,
      childActivities: (ann.childActivities || []).join('; '),
      otherPersonActivities: (ann.otherPersonActivities || []).join('; '),
      locations: (ann.locations || []).join('; ')
    }));

    const csvWriter = createObjectCsvWriter({
      path: filepath,
      header: Object.keys(flattenedAnnotations[0]).map(key => ({ id: key, title: key }))
    });

    await csvWriter.writeRecords(flattenedAnnotations);
    res.download(filepath, filename, (err) => {
      if (err) {
        console.error('Download error:', err);
      }
      // Clean up file after download
      fs.unlinkSync(filepath);
    });
  } catch (error) {
    console.error('Export error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get dropdown options
app.get('/api/options', basicAuth, (req, res) => {
  res.json({
    locations: [
      "balcony", "bathroom", "bedroom", "car", "closet", "deck", "dining room",
      "garage", "garden", "hallway", "kitchen", "laundry room", "living room",
      "office", "outside", "stairway", "storage room", "other"
    ],
    childActivities: [
      "dancing", "drawing", "drinking", "eating", "exploring", "gardening",
      "getting dressed", "looking at device", "music time", "nothing",
      "nursing", "other", "playing", "standing", "walking", "watching tv", "reading"
    ],
    postures: [
      "being held", "crawling", "lying down", "sitting", "walking"
    ],
    otherPersonTypes: ["adult", "child"],
    otherPersonActivities: [
      "cleaning", "cooking", "dancing", "drawing", "drinking", "eating",
      "exploring", "gardening", "getting dressed", "looking at device",
      "music time", "nothing", "nursing", "other", "playing", "standing",
      "walking", "watching tv", "reading"
    ],
    confidenceLevels: ["1", "2", "3"],
    ratingLevels: ["1", "2", "3", "4", "5"]
  });
});

// Start server
app.listen(PORT, '0.0.0.0', () => {
  console.log(`✓ Server running on http://localhost:${PORT}`);
  console.log(`✓ API available at http://localhost:${PORT}/api`);
  console.log(`✓ Experiment at http://localhost:${PORT}/experiment`);
});

module.exports = app;