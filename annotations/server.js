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

// ============================================================================
// SCHEMAS - Activity Annotations
// ============================================================================

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

// ============================================================================
// SCHEMAS - Clip Alignment Annotations with Prolific
// ============================================================================

const ProlificUserSchema = new mongoose.Schema({
  prolificPid: { type: String, required: true, unique: true, index: true },
  annotatorIndex: { type: Number, required: true, index: true },
  mode: { type: String, required: true, enum: ['utterances', 'images'] },
  studyId: String,
  sessionId: String,
  createdAt: { type: Date, default: Date.now }
});

const ClipAlignmentSchema = new mongoose.Schema({
  prolificPid: { type: String, required: true, index: true },
  annotatorIndex: { type: Number, required: true, index: true },
  rowIndex: { type: Number, required: true, index: true },
  mode: { type: String, required: true, enum: ['utterances', 'images'] },
  selectedPosition: { type: Number, required: true },
  correctPosition: { type: Number, required: true },
  isCorrect: { type: Boolean, required: true },
  utterance: String,
  distractorUtt1: String,
  distractorUtt2: String,
  distractorUtt3: String,
  imagePath: String,
  distractorImg1: String,
  distractorImg2: String,
  distractorImg3: String,
  timestamp: { type: Date, default: Date.now }
});

ClipAlignmentSchema.index({ prolificPid: 1, rowIndex: 1, mode: 1 }, { unique: true });

const User = mongoose.model('User', UserSchema);
const Annotation = mongoose.model('Annotation', AnnotationSchema);
const ProlificUser = mongoose.model('ProlificUser', ProlificUserSchema);
const ClipAlignment = mongoose.model('ClipAlignment', ClipAlignmentSchema);

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

// Create directories
const uploadsDir = path.join(__dirname, 'uploads');
const exportsDir = path.join(__dirname, 'exports');
const clipImagesDir = path.join(__dirname, 'clip_images');

[uploadsDir, exportsDir, clipImagesDir].forEach(dir => {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir);
  }
});

// File upload configuration
const csvUpload = multer({ dest: uploadsDir });

// Serve static files
app.use('/experiment', express.static('public'));

// ============================================================================
// ROUTES - General
// ============================================================================

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

// ============================================================================
// ROUTES - Activity Annotations
// ============================================================================

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
    const filepath = path.join(exportsDir, filename);

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

// ============================================================================
// ROUTES - Clip Alignment Annotations with Prolific
// ============================================================================

// Store clip alignment data in memory
let clipAlignmentData = [];

// Load CSV data from file
function loadClipAlignmentCSV(csvPath) {
  clipAlignmentData = [];
  return new Promise((resolve, reject) => {
    fs.createReadStream(csvPath)
      .pipe(csv())
      .on('data', (row) => {
        clipAlignmentData.push({
          annotatorIndex: parseInt(row.annotator_index),
          utterance: row.utterance,
          distractorUtt1: row.distractor_utt1,
          distractorUtt2: row.distractor_utt2,
          distractorUtt3: row.distractor_utt3,
          imagePath: row.image_path,
          distractorImg1: row.distractor_img1,
          distractorImg2: row.distractor_img2,
          distractorImg3: row.distractor_img3
        });
      })
      .on('end', () => {
        console.log(`✓ Loaded ${clipAlignmentData.length} clip alignment items`);
        resolve();
      })
      .on('error', reject);
  });
}

// Load CSV on startup if file exists
const clipAlignmentCSVPath = path.join(__dirname, 'data', 'clip_alignment.csv');
if (fs.existsSync(clipAlignmentCSVPath)) {
  loadClipAlignmentCSV(clipAlignmentCSVPath).catch(err => {
    console.error('Error loading clip alignment CSV:', err);
  });
}

// Upload/reload clip alignment CSV
app.post('/api/clip-alignment/upload-csv', basicAuth, csvUpload.single('csvFile'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    await loadClipAlignmentCSV(req.file.path);
    
    // Clean up uploaded file
    fs.unlinkSync(req.file.path);
    
    res.json({ 
      success: true, 
      count: clipAlignmentData.length,
      message: `Loaded ${clipAlignmentData.length} items`
    });
  } catch (error) {
    console.error('CSV upload error:', error);
    if (req.file && fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path);
    }
    res.status(500).json({ error: error.message });
  }
});

// Register Prolific user and assign annotation index
app.post('/api/clip-alignment/register', async (req, res) => {
  try {
    const { prolificPid, studyId, sessionId } = req.body;
    
    if (!prolificPid) {
      return res.status(400).json({ error: 'Prolific PID is required' });
    }

    // Check if user already exists
    let user = await ProlificUser.findOne({ prolificPid });
    
    if (user) {
      // Return existing assignment
      req.session.prolificPid = user.prolificPid;
      req.session.annotatorIndex = user.annotatorIndex;
      
      return res.json({
        success: true,
        annotatorIndex: user.annotatorIndex,
        mode: user.mode,
        existing: true
      });
    }

    // Find the next available annotation index (0-79)
    const assignedIndices = await ProlificUser.distinct('annotatorIndex');
    let nextIndex = null;
    
    for (let i = 0; i < 80; i++) {
      if (!assignedIndices.includes(i)) {
        nextIndex = i;
        break;
      }
    }
    
    if (nextIndex === null) {
      return res.status(400).json({ 
        error: 'All annotation indices have been assigned (0-79)' 
      });
    }

    // Determine mode based on annotation index (even = images, odd = utterances)
    const mode = nextIndex % 2 === 0 ? 'images' : 'utterances';

    // Create new user
    user = new ProlificUser({
      prolificPid,
      annotatorIndex: nextIndex,
      mode,
      studyId: studyId || 'unknown',
      sessionId: sessionId || 'unknown'
    });
    
    await user.save();

    req.session.prolificPid = user.prolificPid;
    req.session.annotatorIndex = user.annotatorIndex;

    console.log(`✓ Registered new Prolific user: ${prolificPid}, Index: ${nextIndex}, Mode: ${mode}`);

    res.json({
      success: true,
      annotatorIndex: nextIndex,
      mode,
      existing: false
    });
  } catch (error) {
    console.error('Registration error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get clip alignment annotations (load data for experiment)
app.get('/api/clip-alignment/load', async (req, res) => {
  try {
    if (clipAlignmentData.length === 0) {
      return res.status(400).json({ 
        error: 'No clip alignment data loaded. Please upload a CSV file first.' 
      });
    }

    const annotatorIndex = parseInt(req.query.annotator_index);
    if (isNaN(annotatorIndex)) {
      console.log(annotatorIndex)
      return res.status(400).json({ error: 'Valid annotator_index is required' });
    }

    // Filter data by annotation index
    const filteredData = clipAlignmentData.filter(item => 
      item.annotatorIndex === annotatorIndex
    );

    if (filteredData.length === 0) {
      return res.status(400).json({ 
        error: `No data found for annotator_index ${annotatorIndex}` 
      });
    }

    console.log(`✓ Loaded ${filteredData.length} items for annotator_index ${annotatorIndex}`);

    res.json({ 
      success: true,
      annotations: filteredData 
    });
  } catch (error) {
    console.error('Load annotations error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Save clip alignment results
app.post('/api/clip-alignment/results', async (req, res) => {
  try {
    const { results } = req.body;
    
    if (!results || !Array.isArray(results)) {
      return res.status(400).json({ error: 'Invalid results format' });
    }

    const savedResults = [];
    
    for (const result of results) {
      const alignmentData = {
        prolificPid: result.prolific_pid,
        annotatorIndex: result.annotator_index,
        rowIndex: result.row_index,
        mode: result.mode,
        selectedPosition: result.selected_position,
        correctPosition: result.correct_position,
        isCorrect: result.is_correct,
        utterance: result.utterance,
        distractorUtt1: result.distractorUtt1,
        distractorUtt2: result.distractorUtt2,
        distractorUtt3: result.distractorUtt3,
        imagePath: result.imagePath || result.image_path,
        distractorImg1: result.distractorImg1 || result.distractor_img1,
        distractorImg2: result.distractorImg2 || result.distractor_img2,
        distractorImg3: result.distractorImg3 || result.distractor_img3,
        timestamp: new Date(result.timestamp)
      };
      console.log(alignmentData)
      const saved = await ClipAlignment.findOneAndUpdate(
        { 
          prolificPid: result.prolific_pid,
          rowIndex: result.row_index,
          mode: result.mode 
        },
        alignmentData,
        { upsert: true, new: true }
      );
      
      savedResults.push(saved);
    }

    res.json({ 
      success: true, 
      saved: savedResults.length 
    });
  } catch (error) {
    console.error('Save results error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Export clip alignment results
app.get('/api/clip-alignment/export', async (req, res) => {
  try {
    const mode = req.query.mode; // Optional: filter by mode
    const annotatorIndex = req.query.annotator_index; // Optional: filter by index
    
    const query = {};
    if (mode) {
      query.mode = mode;
    }
    if (annotatorIndex !== undefined) {
      query.annotatorIndex = parseInt(annotatorIndex);
    }

    const results = await ClipAlignment.find(query)
      .sort({ annotatorIndex: 1, rowIndex: 1 })
      .lean();

    if (results.length === 0) {
      return res.status(400).json({ error: 'No results to export' });
    }

    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const modeStr = mode ? `_${mode}` : '';
    const indexStr = annotatorIndex !== undefined ? `_idx${annotatorIndex}` : '';
    const filename = `clip_alignment_results${modeStr}${indexStr}_${timestamp}.csv`;
    const filepath = path.join(exportsDir, filename);

    const csvWriter = createObjectCsvWriter({
      path: filepath,
      header: [
        { id: 'prolificPid', title: 'prolific_pid' },
        { id: 'annotatorIndex', title: 'annotator_index' },
        { id: 'rowIndex', title: 'row_index' },
        { id: 'mode', title: 'mode' },
        { id: 'selectedPosition', title: 'selected_position' },
        { id: 'correctPosition', title: 'correct_position' },
        { id: 'isCorrect', title: 'is_correct' },
        { id: 'utterance', title: 'utterance' },
        { id: 'distractorUtt1', title: 'distractor_utt1' },
        { id: 'distractorUtt2', title: 'distractor_utt2' },
        { id: 'distractorUtt3', title: 'distractor_utt3' },
        { id: 'imagePath', title: 'image_path' },
        { id: 'distractorImg1', title: 'distractor_img1' },
        { id: 'distractorImg2', title: 'distractor_img2' },
        { id: 'distractorImg3', title: 'distractor_img3' },
        { id: 'timestamp', title: 'timestamp' }
      ]
    });

    await csvWriter.writeRecords(results);
    
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

// Get Prolific user stats (for admin/debugging)
app.get('/api/clip-alignment/stats', async (req, res) => {
  try {
    const totalUsers = await ProlificUser.countDocuments();
    const usersByMode = await ProlificUser.aggregate([
      { $group: { _id: '$mode', count: { $sum: 1 } } }
    ]);
    const completedAnnotations = await ClipAlignment.aggregate([
      { 
        $group: { 
          _id: { prolificPid: '$prolificPid', mode: '$mode' },
          count: { $sum: 1 }
        } 
      }
    ]);

    res.json({
      success: true,
      totalUsers,
      usersByMode,
      completedAnnotations
    });
  } catch (error) {
    console.error('Stats error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Serve images for clip alignment
app.use('/api/clip-alignment/images', express.static(clipImagesDir));

// ============================================================================
// START SERVER
// ============================================================================

app.listen(PORT, '0.0.0.0', () => {
  console.log(`✓ Server running on http://localhost:${PORT}`);
  console.log(`✓ API available at http://localhost:${PORT}/api`);
  console.log(`✓ Activity Annotations at http://localhost:${PORT}/experiment/index.html`);
  console.log(`✓ Clip Alignment at http://localhost:${PORT}/experiment/clipalignment.html`);
});

module.exports = app;