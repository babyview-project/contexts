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
  primaryActivityOther: String, // for when primaryActivity is "other"
  otherActivitiesOther: [String],
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
  const goldStandardsPath = path.join(__dirname, 'gold_standards.csv');
  
  console.log('Attempting to load gold standards...');
  
  if (!fs.existsSync(goldStandardsPath)) {
    console.warn('⚠ Gold standards CSV file not found at:', goldStandardsPath);
    console.warn('⚠ No gold standards loaded. Training phase will not work.');
    return;
  }

  try {
    const data = await loadCSV(goldStandardsPath, {
      videoFilename: 'video_filename',
      order: 'order',
      primaryActivity: 'primary_activity',
      anyoneInteracting: 'anyone_interacting',
      type: 'type',
      modality: 'modality',
      primaryActivityReasoning: 'pa_reasoning',
      interactingReasoning: 'ai_reasoning'
    });

    // Parse other_activities from semicolon-separated string to array
    data.forEach(item => {
      item.otherActivities = item.otherActivities
        ? item.otherActivities.split(';').map(s => s.trim()).filter(Boolean)
        : [];
    });

    // Split by modality
    const doingData   = data.filter(item => item.modality === 'do');
    const seeingData  = data.filter(item => item.modality === 'see');

    goldStandardDataDoing  = doingData;
    goldStandardDataSeeing = seeingData;

    exampleVideoDataDoing  = doingData.filter(item => item.type === 'example');
    exampleVideoDataSeeing = seeingData.filter(item => item.type === 'example');

    console.log(`✓ Loaded ${doingData.filter(i => i.type === 'gold').length} gold standard DOING videos`);
    console.log(`✓ Loaded ${exampleVideoDataDoing.length} example DOING videos`);
    console.log(`✓ Loaded ${seeingData.filter(i => i.type === 'gold').length} gold standard SEEING videos`);
    console.log(`✓ Loaded ${exampleVideoDataSeeing.length} example SEEING videos`);

    if (goldStandardDataDoing.length === 0 && goldStandardDataSeeing.length === 0) {
      console.warn('⚠ No gold standards loaded. Training phase will not work.');
    }

  } catch (error) {
    console.error('✗ Error loading gold standards CSV:', error);
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

const contextVideosDir = path.join(__dirname, 'sampled_context_videos');
const goldStandardVideosDir = path.join(__dirname, 'goldstandard_videos');

// Create training video directories if they don't exist
[contextVideosDir, goldStandardVideosDir].forEach(dir => {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir);
    console.log(`Created directory: ${dir}`);
  }
});

// Serve training video files
app.use('/training-videos', express.static(contextVideosDir));
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
        error: `No example videos found for task type '${taskType}'. Please check gold_standards.csv has rows with type=example` 
      });
    }
    
    // Verify video files exist
    const validVideos = [];
    for (const vid of exampleVideos) {
      const videoPath1 = path.join(contextVideosDir, vid.videoFilename);
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
        error: `No gold standard videos found for task type '${taskType}'. Please check gold_standards.csv has rows with type=gold` 
      });
    }
    
    // Verify video files exist
    const videos = [];
    for (const gs of goldVideos) {
      const videoPath1 = path.join(contextVideosDir, gs.videoFilename);
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
    const { videoFilenames, taskType } = req.body;
    console.log(taskType)
    const annotatorName = req.session.annotatorName;
    
    if (!annotatorName) {
      return res.status(401).json({ error: 'Not logged in' });
    }

    if (!videoFilenames || !Array.isArray(videoFilenames)) {
      return res.status(400).json({ error: 'Video filenames array required' });
    }
    let existingAnnotations;
    if (taskType == "see") {
          existingAnnotations = await Annotation.find({
          annotatorName,
          videoFilename: { $in: videoFilenames },
          taskType: taskType
        });
    } else {
          existingAnnotations = await Annotation.find({
          annotatorName,
          videoFilename: { $in: videoFilenames },
          taskType: { $ne: 'see' } // exclude 'see' annotations for 'do' task so that we also get blank type entries
    });
    }

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
      primaryActivityOther: req.body.primaryActivityOther || '',
      otherActivitiesOther: req.body.otherActivitiesOther || [],
      anyoneInteracting: req.body.anyoneInteracting,
      isTraining: req.body.isTraining || false,
      attemptNumber: req.body.attemptNumber || '',
      taskType: req.body.taskType || 'do',
      updatedAt: new Date()
    };

    const result = await Annotation.findOneAndUpdate(
      { annotatorName, videoFilename: req.body.videoFilename, taskType: annotationData.taskType },
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
  let activities = []
  if (req.query.taskType == 'see') {
    // no looking around + moving around + postural
    activities = [
      "cleaning", "cooking", "conversing", "crawling", "walking",
      "drawing", "getting dressed", "eating/drinking", "playing with nature",
      "playing with person", "playing with pet", "playing with toy", "playing with household object", 
       "playing with instrument",
       "dancing", "reading time", 
       "screen time", "other"
    ]
  } else {
    // no conversing + gardening
    activities = [
      "being moved around", "crawling", "walking", "cleaning", "looking around", 
      "cooking", "drawing", "eating/drinking", "getting dressed", 
       "reading time", "playing with person", "playing with pet", "playing with instrument",
       "dancing", "playing with toy", "playing with household object", "screen time","other"
    ]
  }
  res.json({
    activities: activities,
    confidenceLevels: ["1", "2", "3"]
  });
});

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