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
  origin: process.env.CLIENT_URL || process.env.PORT,
  credentials: true
}));
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

let server;

try {
  let credentials = process.env.CREDENTIALS_PATH || 'credentials/';
  var privateKey  = fs.readFileSync(`${credentials}ssl_key.pem`),
      certificate = fs.readFileSync(`${credentials}ssl_cert.pem`),
      options     = {key: privateKey, cert: certificate};
  server = require('https').createServer(options, app);
  console.log('✓ Using HTTPS');
} catch (err) {
  console.log("✗ Cannot find SSL certificates; falling back to HTTP");
  server = require('http').createServer(app);
}

// Session configuration
app.use(session({
  secret: process.env.SESSION_SECRET || 'secret-key-temp',
  resave: false,
  saveUninitialized: false,
  store: MongoStore.create({
    mongoUrl: process.env.MONGODB_URI || 'mongodb://localhost:27017/activity_annotations'
  }),
  cookie: {
    maxAge: 1000 * 60 * 60 // 1 hour
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
  primaryActivity: { type: String, required: true },
  primaryActivityConfidence: { type: String, required: true },
  otherActivities: [String],
  otherActivitiesConfidence: { type: String, required: true },
  anyoneInteracting: { type: String, required: true },
  isTraining: { type: Boolean, default: false },  
  attemptNumber: { type: Number },      
  taskType: { type: String, enum: ['do', 'observe'], default: 'do' },          
  updatedAt: { type: Date, default: Date.now }
});

AnnotationSchema.index({ annotatorName: 1, videoFilename: 1, isTraining: 1, attemptNumber: 1 });

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

// Gold standard data loaded from CSV
let goldStandardDataDoing = [];
let goldStandardDataSeeing = [];
let exampleVideoDataDoing = null;
let exampleVideoDataSeeing = null;

let useAuth = process.env.USE_BASIC_AUTH === 'true';


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
// HELPER FUNCTIONS
// ============================================================================

function loadCSV(csvPath, fieldMap) {
  const dataArray = [];

  return new Promise((resolve, reject) => {
    fs.createReadStream(csvPath)
      .pipe(csv())
      .on('data', (row) => {
        const item = {};
        for (const [key, csvCol] of Object.entries(fieldMap)) {
          let value = row[csvCol];
          // automatically parse integers if key includes "Index" or "index"
          if (/index/i.test(key)) value = parseInt(value);
          item[key] = value;
        }
        dataArray.push(item);
      })
      .on('end', () => {
        console.log(`✓ Loaded ${dataArray.length} items from ${csvPath}`);
        resolve(dataArray);
      })
      .on('error', reject);
  });
}

// Load gold standards from CSV
async function loadGoldStandards() {
  const goldStandardsDoingPath = path.join(__dirname, 'gold_standards_doing.csv');
  const goldStandardsSeeingPath = path.join(__dirname, 'gold_standards_seeing.csv');
  
  console.log('Attempting to load gold standards...');
  
  // Load DOING gold standards
  if (fs.existsSync(goldStandardsDoingPath)) {
    try {
      const dataDoing = await loadCSV(goldStandardsDoingPath, {
        videoFilename: 'video_filename',
        primaryActivity: 'primary_activity',
        primaryActivityConfidence: 'primary_activity_confidence',
        otherActivities: 'other_activities',
        otherActivitiesConfidence: 'other_activities_confidence',
        anyoneInteracting: 'anyone_interacting',
        type: 'type',
        primaryActivityReasoning: 'pa_reasoning',
        otherActivitiesReasoning: 'oa_reasoning',
        interactingReasoning: 'ao_reasoning'
      });

      // Parse other_activities from semicolon-separated string to array
      dataDoing.forEach(item => {
        if (item.otherActivities) {
          item.otherActivities = item.otherActivities.split(';').map(s => s.trim()).filter(Boolean);
        } else {
          item.otherActivities = [];
        }
      });

      goldStandardDataDoing = dataDoing;
      exampleVideoDataDoing = dataDoing.filter(item => item.type === 'example');
      
      const exampleCountDoing = dataDoing.filter(item => item.type === 'example').length;
      const goldCountDoing = dataDoing.filter(item => item.type === 'gold').length;
      
      console.log(`✓ Loaded ${goldCountDoing} gold standard DOING videos`);
      console.log(`✓ Loaded ${exampleCountDoing} example DOING videos`);
    } catch (error) {
      console.error('✗ Error loading DOING gold standards CSV:', error);
    }
  } else {
    console.warn('⚠ Gold standards DOING CSV file not found at:', goldStandardsDoingPath);
  }
  
  // Load SEEING gold standards
  if (fs.existsSync(goldStandardsSeeingPath)) {
    try {
      const dataSeeing = await loadCSV(goldStandardsSeeingPath, {
        videoFilename: 'video_filename',
        primaryActivity: 'primary_activity',
        primaryActivityConfidence: 'primary_activity_confidence',
        otherActivities: 'other_activities',
        otherActivitiesConfidence: 'other_activities_confidence',
        anyoneInteracting: 'anyone_interacting',
        type: 'type',
        primaryActivityReasoning: 'pa_reasoning',
        otherActivitiesReasoning: 'oa_reasoning',
        interactingReasoning: 'ao_reasoning'
      });

      // Parse other_activities
      dataSeeing.forEach(item => {
        if (item.otherActivities) {
          item.otherActivities = item.otherActivities.split(';').map(s => s.trim()).filter(Boolean);
        } else {
          item.otherActivities = [];
        }
      });

      goldStandardDataSeeing = dataSeeing;
      exampleVideoDataSeeing = dataSeeing.filter(item => item.type === 'example');
      
      const exampleCountSeeing = dataSeeing.filter(item => item.type === 'example').length;
      const goldCountSeeing = dataSeeing.filter(item => item.type === 'gold').length;
      
      console.log(`✓ Loaded ${goldCountSeeing} gold standard SEEING videos`);
      console.log(`✓ Loaded ${exampleCountSeeing} example SEEING videos`);
    } catch (error) {
      console.error('✗ Error loading SEEING gold standards CSV:', error);
    }
  } else {
    console.warn('⚠ Gold standards SEEING CSV file not found at:', goldStandardsSeeingPath);
  }
  
  if (goldStandardDataDoing.length === 0 && goldStandardDataSeeing.length === 0) {
    console.warn('⚠ No gold standards loaded. Training phase will not work.');
  }
}

// AUTH
const requireAuth = (req, res, next) => {
  console.log('Session ID:', req.sessionID);
  console.log('Authenticated:', req.session.authenticated);
  console.log('Cookie:', req.headers.cookie);
  console.log("Use Auth:", useAuth);
  if (!useAuth) {
    return next();
  }
  
  if (req.session.authenticated) {
    return next();
  }
  
  return res.status(401).json({ error: 'Authentication required' });
};
// Login endpoint
app.post('/api/auth/login', async (req, res) => {
  try {
    const { username, password } = req.body;
    
    if (username === process.env.APP_USERNAME && password === process.env.APP_PASSWORD) {
      req.session.authenticated = true;
      req.session.username = username;
      
      res.json({ 
        success: true,
        message: 'Authentication successful'
      });
    } else {
      res.status(401).json({ error: 'Invalid credentials' });
    }
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Logout endpoint
app.post('/api/auth/logout', (req, res) => {
  req.session.destroy();
  res.json({ success: true });
});

// ============================================================================
// ROUTES - General
// ============================================================================


// Health check
app.get('/api/health', requireAuth, (req, res) => {
  res.json({ 
    status: 'ok', 
    timestamp: new Date(), 
    mongodb: mongoose.connection.readyState === 1,
    goldStandardsLoaded: {
      doingExamples: (exampleVideoDataDoing || []).length,
      doingGold: (goldStandardDataDoing || []).filter(v => v.type === 'gold').length,
      seeingExamples: (exampleVideoDataSeeing || []).length,
      seeingGold: (goldStandardDataSeeing || []).filter(v => v.type === 'gold').length
    }
  });
});

// Login/Create User
app.post('/api/login', requireAuth, async (req, res) => {
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

const sampledVideosDir = path.join(__dirname, 'sampled_context_videos');

// ============================================================================
// ROUTES - Training Videos
// ============================================================================

const exampleVideosDir = path.join(__dirname, 'example_videos');
const goldStandardVideosDir = path.join(__dirname, 'goldstandard_videos');

// Create training video directories if they don't exist
[exampleVideosDir, goldStandardVideosDir].forEach(dir => {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir);
    console.log(`Created directory: ${dir}`);
  }
});

// Serve training video files
app.use('/training-videos', express.static(exampleVideosDir));
app.use('/training-videos', express.static(goldStandardVideosDir));

// Get example video (the one video shown with annotations)
app.get('/api/training/example-video', requireAuth, async (req, res) => {
  try {
    const taskType = req.query.taskType || 'do';
    const exampleVideos = taskType === 'see' 
      ? (goldStandardDataSeeing || []).filter(item => item.type === 'example')
      : (goldStandardDataDoing || []).filter(item => item.type === 'example');
    
    if (exampleVideos.length === 0) {
      return res.status(404).json({ 
        error: `No example videos found for task type '${taskType}'. Please check gold_standards_${taskType === 'see' ? 'seeing' : 'doing'}.csv has rows with type=example` 
      });
    }
    
    // Verify video files exist
    const validVideos = [];
    for (const vid of exampleVideos) {
      const videoPath1 = path.join(exampleVideosDir, vid.videoFilename);
      const videoPath2 = path.join(goldStandardVideosDir, vid.videoFilename);
      
      if (fs.existsSync(videoPath1) || fs.existsSync(videoPath2)) {
        validVideos.push({
          videoFilename: vid.videoFilename,
          description: vid.description || '',
          primaryActivity: vid.primaryActivity,
          primaryActivityConfidence: vid.primaryActivityConfidence,
          otherActivities: vid.otherActivities || [],
          otherActivitiesConfidence: vid.otherActivitiesConfidence,
          anyoneInteracting: vid.anyoneInteracting,
          primaryActivityReasoning: vid.primaryActivityReasoning,
          otherActivitiesReasoning: vid.otherActivitiesReasoning,
          interactingReasoning: vid.interactingReasoning
        });
      } else {
        console.warn(`⚠ Example video file not found: ${vid.videoFilename}`);
      }
    }
    
    if (validVideos.length === 0) {
      return res.status(404).json({ 
        error: `Example video files not found for task type '${taskType}'`,
        expectedVideos: exampleVideos.map(v => v.videoFilename)
      });
    }
    
    console.log(`Returning ${validVideos.length} example videos for task type '${taskType}'`);
    
    res.json({
      success: true,
      videos: validVideos,
      count: validVideos.length
    });
  } catch (error) {
    console.error('Example video error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get gold standard videos (for testing)
app.get('/api/training/gold-standard-videos', requireAuth, async (req, res) => {
  try {
    const taskType = req.query.taskType || 'do';
    const goldVideos = taskType === 'see'
      ? (goldStandardDataSeeing || []).filter(item => item.type === 'gold')
      : (goldStandardDataDoing || []).filter(item => item.type === 'gold');
    
    if (goldVideos.length === 0) {
      return res.status(404).json({ 
        error: `No gold standard videos found for task type '${taskType}'. Please check gold_standards_${taskType === 'see' ? 'seeing' : 'doing'}.csv has rows with type=gold` 
      });
    }
    
    // Verify video files exist
    const videos = [];
    for (const gs of goldVideos) {
      const videoPath1 = path.join(exampleVideosDir, gs.videoFilename);
      const videoPath2 = path.join(goldStandardVideosDir, gs.videoFilename);
      
      if (fs.existsSync(videoPath1) || fs.existsSync(videoPath2)) {
        videos.push({
          videoFilename: gs.videoFilename,
          description: gs.description || '',
          primaryActivity: gs.primaryActivity,
          primaryActivityConfidence: gs.primaryActivityConfidence,
          otherActivities: gs.otherActivities || [],
          otherActivitiesConfidence: gs.otherActivitiesConfidence,
          anyoneInteracting: gs.anyoneInteracting,
          type: gs.type
        });
      } else {
        console.warn(`⚠ Gold standard video file not found: ${gs.videoFilename}`);
      }
    }
    
    if (videos.length === 0) {
      return res.status(404).json({ 
        error: `No gold standard video files found for task type '${taskType}'`,
        expectedVideos: goldVideos.map(gs => gs.videoFilename)
      });
    }
    
    console.log(`Returning ${videos.length} gold standard videos for task type '${taskType}'`);
    
    res.json({
      success: true,
      videos: videos,
      count: videos.length
    });
  } catch (error) {
    console.error('Gold standard videos error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Reload gold standards from CSV
app.post('/api/training/reload-gold-standards', requireAuth, async (req, res) => {
  try {
    await loadGoldStandards();
    
    res.json({
      success: true,
      exampleCount: exampleVideoData ? 1 : 0,
      goldCount: goldStandardData.length,
      message: 'Gold standards reloaded from CSV'
    });
  } catch (error) {
    console.error('Reload gold standards error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Export current gold standards (in case you want to see what's loaded)
app.get('/api/training/export-gold-standards', requireAuth, async (req, res) => {
  try {
    const allData = [];
    
    if (exampleVideoData) {
      allData.push({ ...exampleVideoData, type: 'example' });
    }
    
    goldStandardData.forEach(gs => {
      allData.push({ ...gs, type: 'gold' });
    });
    
    if (allData.length === 0) {
      return res.status(400).json({ error: 'No gold standards loaded' });
    }
    
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `gold_standards_export_${timestamp}.csv`;
    const filepath = path.join(exportsDir, filename);
    
    // Flatten arrays to semicolon-separated strings
    const flattenedData = allData.map(gs => ({
      video_filename: gs.videoFilename,
      primary_activity: gs.primaryActivity,
      primary_activity_confidence: gs.primaryActivityConfidence,
      other_activities: (gs.otherActivities || []).join('; '),
      other_activities_confidence: gs.otherActivitiesConfidence,
      anyone_interacting: gs.anyoneInteracting,
      type: gs.type
    }));
    
    const csvWriter = createObjectCsvWriter({
      path: filepath,
      header: [
        { id: 'video_filename', title: 'video_filename' },
        { id: 'primary_activity', title: 'primary_activity' },
        { id: 'primary_activity_confidence', title: 'primary_activity_confidence' },
        { id: 'other_activities', title: 'other_activities' },
        { id: 'other_activities_confidence', title: 'other_activities_confidence' },
        { id: 'anyone_interacting', title: 'anyone_interacting' },
        { id: 'type', title: 'type' }
      ]
    });
    
    await csvWriter.writeRecords(flattenedData);
    
    res.download(filepath, filename, (err) => {
      if (err) {
        console.error('Download error:', err);
      }
      // Clean up file after download
      fs.unlinkSync(filepath);
    });
  } catch (error) {
    console.error('Export gold standards error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Serve video files
app.use('/videos', express.static(sampledVideosDir));

// Get list of videos
app.get('/api/video-list', requireAuth, (req, res) => {
  try {
    if (!fs.existsSync(sampledVideosDir)) {
      return res.status(404).json({ error: 'Video directory not found' });
    }
    
    const files = fs.readdirSync(sampledVideosDir);
    const videoExtensions = ['.mp4', '.webm', '.ogg', '.mov', '.avi'];
    const videoFiles = files
      .filter(file => videoExtensions.some(ext => file.toLowerCase().endsWith(ext)))
      .sort(); // Sort alphabetically
    
    res.json({ 
      success: true, 
      videos: videoFiles,
      count: videoFiles.length 
    });
  } catch (error) {
    console.error('Error reading video directory:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get current user
app.get('/api/user', requireAuth, (req, res) => {
  if (req.session.annotatorName) {
    res.json({ name: req.session.annotatorName });
  } else {
    res.status(401).json({ error: 'Not logged in' });
  }
});

// Logout
app.post('/api/logout', requireAuth, (req, res) => {
  req.session.destroy();
  res.json({ success: true });
});

// ============================================================================
// ROUTES - Activity Annotations
// ============================================================================

// Get existing annotations for video list
app.post('/api/get-annotations-for-videos', requireAuth, async (req, res) => {
  try {
    const { videoFilenames } = req.body;
    const annotatorName = req.session.annotatorName;
    
    if (!annotatorName) {
      return res.status(401).json({ error: 'Not logged in' });
    }

    if (!videoFilenames || !Array.isArray(videoFilenames)) {
      return res.status(400).json({ error: 'Video filenames array required' });
    }

    const existingAnnotations = await Annotation.find({
      annotatorName,
      videoFilename: { $in: videoFilenames }
    });

    // Create a map of existing annotations
    const annotationMap = {};
    existingAnnotations.forEach(ann => {
      annotationMap[ann.videoFilename] = ann;
    });
    
    res.json({ 
      success: true, 
      annotations: annotationMap
    });
  } catch (error) {
    console.error('Get annotations error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Get existing annotations for a video
app.get('/api/annotations/:videoFilename', requireAuth, async (req, res) => {
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
app.post('/api/annotations', requireAuth, async (req, res) => {
  try {
    const annotatorName = req.session.annotatorName;
    if (!annotatorName) {
      return res.status(401).json({ error: 'Not logged in' });
    }

    const annotationData = {
      annotatorName,
      videoFilename: req.body.videoFilename,
      description: req.body.description || '',
      primaryActivity: req.body.primaryActivity,
      primaryActivityConfidence: req.body.primaryActivityConfidence,
      otherActivities: req.body.otherActivities || [],
      otherActivitiesConfidence: req.body.otherActivitiesConfidence,
      anyoneInteracting: req.body.anyoneInteracting,
      isTraining: req.body.isTraining || false,
      attemptNumber: req.body.attemptNumber || '',
      taskType: req.body.taskType || 'do',
      updatedAt: new Date()
    };

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
app.get('/api/annotations', requireAuth, async (req, res) => {
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

    // Get both training and main annotations
    const annotations = await Annotation.find({ annotatorName })
      .sort({ isTraining: 1, attemptNumber: 1, updatedAt: -1 })
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
      otherActivities: (ann.otherActivities || []).join('; '),
      isTraining: ann.isTraining || false,
      attemptNumber: ann.attemptNumber || ''
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
app.get('/api/options', requireAuth, (req, res) => {
  res.json({
    activities: [
      "cleaning", "cooking", "conversing", "drawing", "drinking", "gardening", 
      "getting dressed", "looking around", "meal time", "moving around",
      "music time", "reading time", "playing", "screen time", "other"
    ],
    confidenceLevels: ["1", "2", "3"]
  });
});

// ============================================================================
// ROUTES - Clip Alignment Annotations with Prolific
// ============================================================================

// Store clip alignment data in memory
let clipAlignmentData = [];

// Catch trials
let catchTrials = [];

const catchTrialFieldMap = {
  utterance: 'utterance',
  distractorUtt1: 'distractor_utt1',
  distractorUtt2: 'distractor_utt2',
  distractorUtt3: 'distractor_utt3',
  imagePath: 'image_path',
  distractorImg1: 'distractor_img1',
  distractorImg2: 'distractor_img2',
  distractorImg3: 'distractor_img3',
  annotatorIndex: 'annotator_index'
};

const clipAlignmentFieldMap = {
  ...catchTrialFieldMap
};

// Load CSV on startup if file exists
const clipAlignmentCSVPath = path.join(__dirname, 'data', 'clip_alignment.csv');
const catchTrialsCSVPath = path.join(__dirname, 'data', 'catch_trials.csv');
if (fs.existsSync(clipAlignmentCSVPath)) {
  loadCSV(clipAlignmentCSVPath, clipAlignmentFieldMap)
    .then(loadedClipAlignmentData => {
      clipAlignmentData = loadedClipAlignmentData;
      console.log(`Loaded ${clipAlignmentData.length} clip alignment items`);
    })
    .catch(err => {
      console.error('Error loading clip alignment CSV:', err);
    });
}

if (fs.existsSync(catchTrialsCSVPath)) {
  loadCSV(catchTrialsCSVPath, catchTrialFieldMap)
    .then(catchTrialsData => {
      catchTrials = catchTrialsData;
      console.log(`Loaded ${catchTrials.length} catch trial items`);
    })
    .catch(err => {
      console.error('Error loading catch trials CSV:', err);
    });
}

// Upload/reload clip alignment CSV
app.post('/api/clip-alignment/upload-csv', requireAuth, csvUpload.single('csvFile'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    clipAlignmentData = await loadCSV(req.file.path, clipAlignmentFieldMap);
    
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
    let testRun = false;
    if (nextIndex === null) {
      if (prolificPid != "test" && prolificPid != "images" && prolificPid != "utterances" && prolificPid != "test_utterances") {
        return res.status(400).json({ 
          error: 'All annotation indices have been assigned (0-79)' 
        });
      } else {
        testRun = true;
        nextIndex = 1;
      }
    }

    // Determine mode based on annotation index (even = images, odd = utterances)
    let mode = nextIndex % 2 === 0 ? 'images' : 'utterances';

    if (prolificPid == "test" || prolificPid == "images") {
      mode = "images";
    } else if (prolificPid == "utterances" || prolificPid == "test_utterances") {
      mode = "utterances";
    }
    if (testRun && prolificPid.startsWith("test")) {
      nextIndex = 1;
    }

    // Create new user
    user = new ProlificUser({
      prolificPid,
      annotatorIndex: nextIndex,
      mode,
      studyId: studyId || 'unknown',
      sessionId: sessionId || 'unknown'
    });
    if (!testRun) {
      await user.save();
    }
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

    // Start with a copy of filteredData
    const combinedData = [...filteredData];

    // Insert all catch trials at random positions
    catchTrials.forEach(catchTrial => {
      const randomIndex = Math.floor(Math.random() * (combinedData.length + 1));
      combinedData.splice(randomIndex, 0, catchTrial);
    });

    console.log(`Combined data length: ${combinedData.length}`);
    console.log(`✓ Loaded ${filteredData.length} items for annotator_index ${annotatorIndex}`);

    res.json({ 
      success: true,
      annotations: combinedData, 
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
      if (catchTrials.map(ct => ct.imagePath).includes(result.imagePath || result.image_path)) {
        result.mode = result.mode + "_AG";
      }
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
    const mode = req.query.mode;
    const annotatorIndex = req.query.annotator_index;
    
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

// Load gold standards before starting server
loadGoldStandards().then(() => {
  server.listen(PORT, '0.0.0.0', () => {
    const protocol = server instanceof require('https').Server ? 'https' : 'http';
    console.log(`✓ Server running on ${protocol}://localhost:${PORT}`);
    console.log(`✓ API available at ${protocol}://localhost:${PORT}/api`);
    console.log(`✓ Activity Annotations at ${protocol}://localhost:${PORT}/experiment/activities.html`);
    console.log(`✓ Clip Alignment at ${protocol}://localhost:${PORT}/experiment/clipalignment.html`);
  });
}).catch(err => {
  console.error('Failed to load gold standards, starting server anyway:', err);
  server.listen(PORT, '0.0.0.0', () => {
    const protocol = server instanceof require('https').Server ? 'https' : 'http';
    console.log(`✓ Server running on ${protocol}://localhost:${PORT}`);
    console.log(`⚠ Warning: Gold standards may not be loaded`);
  });
});

module.exports = app;