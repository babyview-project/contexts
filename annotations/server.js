// server.js - Node.js Express backend for activity annotations
// Run modes (see README.md): --test (no videos/MongoDB needed), --dev (skip controls)

require('dotenv').config();
const express = require('express');
const cors = require('cors');
const session = require('express-session');
const { createObjectCsvStringifier } = require('csv-writer');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const config = require('./lib/config');
const { createMemoryStore, createMongoStore } = require('./lib/store');
const { loadGoldStandards, practiceSet, grade, readCSV, readCSVWithHeader, writeCSVWithBackup } = require('./lib/gold');
const { authenticate, createLoginLimiter, readUsers } = require('./lib/users');
const { asCurrent, labelMap } = require('./lib/labels');

const app = express();
const PORT = process.env.PORT || 3000;
const TASK_TYPES = ['do', 'see'];

let store;
let goldStandards;

if (process.env.CLIENT_URL) {
  app.use(cors({ origin: process.env.CLIENT_URL, credentials: true }));
}
app.use(express.json({ limit: '1mb' }));

// The session store depends on the storage backend, so it is configured in start()
let sessionMiddleware;
app.use((req, res, next) => sessionMiddleware(req, res, next));

let server;
try {
  const credentials = process.env.CREDENTIALS_PATH || 'credentials/';
  const options = {
    key: fs.readFileSync(`${credentials}ssl_key.pem`),
    cert: fs.readFileSync(`${credentials}ssl_cert.pem`)
  };
  server = require('https').createServer(options, app);
  console.log('✓ Using HTTPS');
} catch (err) {
  console.log('✗ Cannot find SSL certificates; falling back to HTTP');
  server = require('http').createServer(app);
}

// ============================================================================
// HELPERS
// ============================================================================

const asyncRoute = (fn) => (req, res) =>
  fn(req, res).catch(error => {
    console.error(`${req.method} ${req.path} error:`, error);
    res.status(500).json({ error: error.message });
  });

const usingAccounts = config.authMode === 'accounts';

const requireAuth = (req, res, next) => {
  if (!usingAccounts || req.session.user) return next();
  return res.status(401).json({ error: 'Login required' });
};

const requireAdmin = (req, res, next) => {
  if (!usingAccounts) return next();
  if (!req.session.user) return res.status(401).json({ error: 'Login required' });
  if (!req.session.user.admin) return res.status(403).json({ error: 'Admin access required' });
  next();
};

// Dev mode for one request: always when the server runs with --dev; otherwise the page can ask for it
// (activities.html?dev=1 sends X-Dev-Mode: 1), which is honored only for admin accounts, or for
// anyone when there are no accounts. Coders can't use it to skip training.
const isDev = (req) => {
  if (config.devMode) return true;
  if (req.get('X-Dev-Mode') !== '1') return false;
  return usingAccounts ? !!req.session.user?.admin : true;
};

// With accounts the coder ID is the logged-in username; otherwise the client sends the typed ID
const requireAnnotator = (req, res, next) => {
  if (usingAccounts) {
    if (!req.session.user) return res.status(401).json({ error: 'Login required' });
    req.annotatorName = req.session.user.username;
    return next();
  }
  let name;
  try {
    name = decodeURIComponent(req.get('X-Annotator-Id') || '').trim();
  } catch (err) {
    return res.status(400).json({ error: 'Invalid annotator ID' });
  }
  if (!name) return res.status(400).json({ error: 'Annotator ID required' });
  if (name.length > 100) return res.status(400).json({ error: 'Annotator ID too long' });
  req.annotatorName = name;
  next();
};

const getTaskType = (req) => {
  const taskType = req.query.taskType || req.body?.taskType;
  return TASK_TYPES.includes(taskType) ? taskType : null;
};

const requireTaskType = (req, res, next) => {
  req.taskType = getTaskType(req);
  if (!req.taskType) return res.status(400).json({ error: "taskType must be 'do' or 'see'" });
  next();
};

const VIDEO_EXTENSIONS = ['.mp4', '.webm', '.ogg', '.mov'];

// The pool of main-task clips (shared by do and see)
async function getMainVideoPool() {
  let videos = [];
  const { mainListFile, mainDir } = config.videos;
  if (mainListFile && !fs.existsSync(mainListFile)) {
    console.warn(`⚠ videos.mainListFile not found: ${mainListFile}`);
  } else if (mainListFile) {
    if (mainListFile.endsWith('.csv')) {
      videos = (await readCSV(mainListFile)).map(r => (r.video_filename || '').trim());
    } else {
      videos = fs.readFileSync(mainListFile, 'utf8').split('\n').map(s => s.trim());
    }
  } else if (fs.existsSync(mainDir)) {
    videos = fs.readdirSync(mainDir).filter(f => VIDEO_EXTENSIONS.some(ext => f.toLowerCase().endsWith(ext)));
  }
  videos = [...new Set(videos.filter(Boolean))].sort();

  if (videos.length === 0 && config.testMode) {
    videos = Array.from({ length: config.test.fakeVideoCount },
      (_, i) => `test_clip_${String(i + 1).padStart(3, '0')}.mp4`);
  }
  return videos;
}

function shuffle(items) {
  const a = [...items];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// Serializes assignment creation so two coders starting together can't overfill a clip
let assignmentLock = Promise.resolve();
function withAssignmentLock(fn) {
  const run = assignmentLock.then(fn, fn);
  assignmentLock = run.catch(() => {});
  return run;
}

// Assign each coder `videosPerCoder` clips, preferring the clips with the fewest coders so far,
// and never giving a clip to more than `ratingsPerClip` coders. null videosPerCoder = all clips.
async function getOrCreateAssignment(annotatorName, taskType) {
  return withAssignmentLock(async () => {
    const existing = await store.getAssignment(annotatorName, taskType);
    if (existing) return existing;

    const pool = await getMainVideoPool();
    const { videosPerCoder, ratingsPerClip } = config.assignment;
    let videos;

    if (!videosPerCoder) {
      videos = shuffle(pool);
    } else {
      const counts = new Map(pool.map(v => [v, 0]));
      for (const a of await store.listAssignments(taskType)) {
        for (const v of a.videos) if (counts.has(v)) counts.set(v, counts.get(v) + 1);
      }
      const candidates = shuffle(pool)
        .filter(v => counts.get(v) < ratingsPerClip)
        .sort((a, b) => counts.get(a) - counts.get(b));
      videos = shuffle(candidates.slice(0, videosPerCoder));
      if (videos.length < videosPerCoder) {
        console.warn(`⚠ ${annotatorName} (${taskType}) assigned only ${videos.length}/${videosPerCoder} clips: ` +
          `remaining clips already have ${ratingsPerClip} coders`);
      }
    }

    if (videos.length === 0) return { annotatorName, taskType, videos: [] };
    console.log(`Assigned ${videos.length} ${taskType} clips to ${annotatorName}`);
    return store.createAssignment({ annotatorName, taskType, videos });
  });
}

// Dev-mode stand-in for an assignment, never stored. The same user + seed always gives the same
// clips; the page's "Shuffle preview clips" button just picks a new seed.
const previewSeed = (req) => String(req.query.previewSeed || req.body?.previewSeed || '').slice(0, 32);

async function previewAssignment(annotatorName, taskType, seed = '') {
  const pool = await getMainVideoPool();
  const n = config.assignment.videosPerCoder || pool.length;
  const key = (v) => crypto.createHash('sha1').update(`${annotatorName}|${taskType}|${seed}|${v}`).digest('hex');
  return pool.map(v => [key(v), v]).sort((a, b) => a[0].localeCompare(b[0])).slice(0, n).map(([, v]) => v);
}

function summarizeTraining(attempts) {
  const { maxAttempts } = config.training;
  return {
    attemptsUsed: attempts.length,
    maxAttempts,
    attemptsRemaining: Math.max(0, maxAttempts - attempts.length),
    passed: attempts.some(a => a.passed),
    bestScore: attempts.length ? Math.max(...attempts.map(a => a.score)) : null,
    lastAttemptAt: attempts.length ? attempts[attempts.length - 1].createdAt : null
  };
}

async function trainingStatus(annotatorName, taskType) {
  return summarizeTraining(await store.listTrainingAttempts({ annotatorName, taskType }));
}

const ANNOTATION_CSV_FIELDS = ['annotatorName', 'taskType', 'videoFilename', 'isTraining', 'attemptNumber',
  'primaryActivity', 'primaryActivityOther', 'anyoneInteracting', 'updatedAt'];

function sendAnnotationsCSV(res, annotations, filename) {
  const stringifier = createObjectCsvStringifier({
    header: ANNOTATION_CSV_FIELDS.map(id => ({ id, title: id }))
  });
  const rows = annotations.map(a => ({
    ...a,
    taskType: a.taskType || 'do',
    isTraining: !!a.isTraining,
    attemptNumber: a.attemptNumber ?? '',
    updatedAt: a.updatedAt ? new Date(a.updatedAt).toISOString() : ''
  }));
  res.setHeader('Content-Type', 'text/csv');
  res.setHeader('Content-Disposition', `attachment; filename="${filename}"`);
  res.send(stringifier.getHeaderString() + stringifier.stringifyRecords(rows));
}

const timestamp = () => new Date().toISOString().replace(/[:.]/g, '-');

// ============================================================================
// ROUTES - Static files and videos
// ============================================================================

app.get('/', (req, res) => res.redirect('/experiment/activities.html'));
app.use('/experiment', express.static(path.join(__dirname, 'public')));
app.use('/videos', requireAuth, express.static(config.videos.mainDir));
config.videos.trainingDirs.forEach(dir => app.use('/training-videos', requireAuth, express.static(dir)));

// ============================================================================
// ROUTES - Auth and config
// ============================================================================

// Per-username lockout is strict; the per-IP limit is looser since coders may share a network
const userLimiter = createLoginLimiter({ maxFailures: 10 });
const ipLimiter = createLoginLimiter({ maxFailures: 50 });

app.post('/api/auth/login', (req, res) => {
  if (!usingAccounts) return res.status(400).json({ error: 'Accounts are not enabled' });
  const username = String(req.body.username || '').trim();
  const userKey = username.toLowerCase();
  if (userLimiter.isBlocked(userKey) || ipLimiter.isBlocked(req.ip)) {
    return res.status(429).json({ error: 'Too many failed attempts. Try again in 15 minutes.' });
  }

  const user = authenticate(config.usersPath, username, req.body.password);
  if (!user) {
    userLimiter.recordFailure(userKey);
    ipLimiter.recordFailure(req.ip);
    return res.status(401).json({ error: 'Incorrect username or password' });
  }
  userLimiter.reset(userKey);
  // New session ID on login (prevents session fixation)
  req.session.regenerate(err => {
    if (err) return res.status(500).json({ error: 'Could not start session' });
    req.session.user = user;
    res.json({ success: true, user });
  });
});

app.post('/api/auth/logout', (req, res) => {
  req.session.destroy(() => res.json({ success: true }));
});

// devMode: whether dev mode is on for this request (see isDev)
app.get('/api/auth/me', (req, res) => {
  res.json({ user: usingAccounts ? (req.session.user || null) : null, devMode: isDev(req) });
});

// Public: everything the client needs before login
app.get('/api/config', (req, res) => {
  res.json({
    authMode: config.authMode,
    devMode: config.devMode,
    testMode: config.testMode,
    training: config.training,
    assignment: config.assignment,
    activities: config.activities
  });
});

app.get('/api/health', requireAdmin, (req, res) => {
  res.json({
    status: 'ok',
    timestamp: new Date(),
    store: store.kind,
    goldStandards: Object.fromEntries(TASK_TYPES.map(t => [t, {
      examples: goldStandards[t].examples.length,
      gold: goldStandards[t].gold.length
    }])),
    goldStandardWarnings: goldStandards.warnings
  });
});

app.post('/api/login', requireAuth, requireAnnotator, asyncRoute(async (req, res) => {
  const user = await store.upsertUser(req.annotatorName);
  res.json({ success: true, user: { name: user.name, createdAt: user.createdAt } });
}));

// ============================================================================
// ROUTES - Training
// ============================================================================

app.get('/api/training/examples', requireAuth, requireTaskType, (req, res) => {
  const videos = goldStandards[req.taskType].examples.map(e => ({
    videoFilename: e.videoFilename,
    primaryActivity: e.primaryActivity,
    anyoneInteracting: e.anyoneInteracting,
    primaryActivityReasoning: e.primaryActivityReasoning,
    interactingReasoning: e.interactingReasoning
  }));
  res.json({ success: true, videos });
});

app.get('/api/training/status', requireAuth, requireAnnotator, requireTaskType, asyncRoute(async (req, res) => {
  res.json({ success: true, ...(await trainingStatus(req.annotatorName, req.taskType)) });
}));

// Practice clips for an attempt; answers are only returned after grading
app.get('/api/training/practice-set', requireAuth, requireAnnotator, requireTaskType, asyncRoute(async (req, res) => {
  const attemptNumber = parseInt(req.query.attempt, 10);
  if (!(attemptNumber >= 1)) return res.status(400).json({ error: 'attempt must be a positive integer' });

  const videos = practiceSet(goldStandards[req.taskType].gold, {
    annotatorName: req.annotatorName,
    taskType: req.taskType,
    attemptNumber,
    videosPerAttempt: config.training.videosPerAttempt
  }).map(g => ({ videoFilename: g.videoFilename }));

  if (videos.length === 0) {
    return res.status(404).json({ error: `No gold standard clips available for '${req.taskType}'` });
  }
  res.json({ success: true, attemptNumber, videos });
}));

// Grade a practice attempt and record it
app.post('/api/training/attempts', requireAuth, requireAnnotator, requireTaskType, asyncRoute(async (req, res) => {
  const { annotatorName, taskType } = req;
  const attemptNumber = parseInt(req.body.attemptNumber, 10);
  const responses = Array.isArray(req.body.responses) ? req.body.responses : [];

  // Dev mode is a sandbox: grade and show feedback, but record nothing
  if (isDev(req)) {
    const graded = grade(goldStandards[taskType].gold, responses);
    if (!graded || graded.nTotal === 0) {
      return res.status(400).json({ error: 'Responses do not match the gold standard clips' });
    }
    const passed = graded.score >= config.training.passThreshold;
    const { maxAttempts } = config.training;
    return res.json({
      success: true, devMode: true, attemptNumber, passed, ...graded,
      status: { attemptsUsed: attemptNumber, maxAttempts, attemptsRemaining: Math.max(0, maxAttempts - attemptNumber), passed }
    });
  }

  const previous = await store.listTrainingAttempts({ annotatorName, taskType });
  if (!(attemptNumber >= 1) || attemptNumber > previous.length + 1) {
    return res.status(400).json({ error: `Expected attempt number ${previous.length + 1}` });
  }
  if (attemptNumber > config.training.maxAttempts) {
    return res.status(403).json({ error: 'No training attempts remaining' });
  }

  const graded = grade(goldStandards[taskType].gold, responses);
  if (!graded || graded.nTotal === 0) {
    return res.status(400).json({ error: 'Responses do not match the gold standard clips' });
  }
  const passed = graded.score >= config.training.passThreshold;

  for (const r of graded.results) {
    await store.saveAnnotation({
      annotatorName, taskType, isTraining: true, attemptNumber,
      videoFilename: r.videoFilename,
      ...r.response
    });
  }
  await store.saveTrainingAttempt({
    annotatorName, taskType, attemptNumber, passed,
    nCorrect: graded.nCorrect, nTotal: graded.nTotal, score: graded.score,
    results: graded.results.map(r => ({
      videoFilename: r.videoFilename, correct: r.correct,
      primaryCorrect: r.primaryCorrect, interactingCorrect: r.interactingCorrect
    }))
  });

  res.json({ success: true, attemptNumber, passed, ...graded, status: await trainingStatus(annotatorName, taskType) });
}));

app.post('/api/training/reload-gold-standards', requireAdmin, asyncRoute(async (req, res) => {
  await reloadGoldStandards();
  res.json({ success: true, warnings: goldStandards.warnings });
}));

// ============================================================================
// ROUTES - Main annotation task
// ============================================================================

// The coder's assigned clips plus whatever they've already annotated
app.get('/api/assignment', requireAuth, requireAnnotator, requireTaskType, asyncRoute(async (req, res) => {
  const { annotatorName, taskType } = req;
  // Dev mode: a preview set that is never stored, so it uses no coder slots and has no saved progress
  if (isDev(req)) {
    return res.json({ success: true, devMode: true, videos: await previewAssignment(annotatorName, taskType, previewSeed(req)), annotations: {} });
  }
  if (!(await trainingStatus(annotatorName, taskType)).passed) {
    return res.status(403).json({ error: 'Training must be passed before the main task' });
  }

  const assignment = await getOrCreateAssignment(annotatorName, taskType);
  const done = await store.findAnnotations({ annotatorName, taskType, isTraining: false });
  const assigned = new Set(assignment.videos);
  const annotations = {};
  done.filter(a => assigned.has(a.videoFilename)).forEach(a => {
    annotations[a.videoFilename] = {
      primaryActivity: a.primaryActivity,
      primaryActivityOther: a.primaryActivityOther || '',
      anyoneInteracting: a.anyoneInteracting
    };
  });
  res.json({ success: true, videos: assignment.videos, annotations });
}));

app.post('/api/annotations', requireAuth, requireAnnotator, requireTaskType, asyncRoute(async (req, res) => {
  const { videoFilename, primaryActivity, primaryActivityOther, anyoneInteracting } = req.body;
  const activities = config.activities[req.taskType];

  if (!activities.includes(primaryActivity)) {
    return res.status(400).json({ error: `Unknown activity '${primaryActivity}'` });
  }
  if (!['yes', 'no'].includes(anyoneInteracting)) {
    return res.status(400).json({ error: 'anyoneInteracting must be yes or no' });
  }
  // Dev mode: validate the answer like a real save, but store nothing
  if (isDev(req)) {
    if (!(await previewAssignment(req.annotatorName, req.taskType, previewSeed(req))).includes(videoFilename)) {
      return res.status(400).json({ error: 'This clip is not in your dev preview set' });
    }
    return res.json({ success: true, devMode: true });
  }
  const assignment = await store.getAssignment(req.annotatorName, req.taskType);
  if (!assignment || !assignment.videos.includes(videoFilename)) {
    return res.status(400).json({ error: 'This clip is not assigned to you' });
  }

  const annotation = await store.saveAnnotation({
    annotatorName: req.annotatorName,
    taskType: req.taskType,
    isTraining: false,
    attemptNumber: null,
    videoFilename,
    primaryActivity,
    primaryActivityOther: primaryActivity === 'other' ? (primaryActivityOther || '').trim() : '',
    anyoneInteracting
  });
  res.json({ success: true, annotation });
}));

// The coder's own annotations as CSV
app.get('/api/export', requireAuth, requireAnnotator, asyncRoute(async (req, res) => {
  const annotations = await store.findAnnotations({ annotatorName: req.annotatorName });
  if (annotations.length === 0) return res.status(400).json({ error: 'No annotations to export' });
  sendAnnotationsCSV(res, annotations, `annotations_${req.annotatorName}_${timestamp()}.csv`);
}));

// ============================================================================
// ROUTES - Admin (training status and coverage)
// ============================================================================

app.get('/api/admin/status', requireAdmin, asyncRoute(async (req, res) => {
  const pool = await getMainVideoPool();
  const users = await store.listUsers();
  const names = new Set(users.map(u => u.name));
  const coders = {};
  const coverage = {};

  for (const taskType of TASK_TYPES) {
    const attempts = await store.listTrainingAttempts({ taskType });
    const assignments = await store.listAssignments(taskType);
    const annotations = await store.findAnnotations({ taskType, isTraining: false });
    attempts.forEach(a => names.add(a.annotatorName));
    assignments.forEach(a => names.add(a.annotatorName));

    const doneByCoder = new Map();
    const ratingsByClip = new Map(pool.map(v => [v, 0]));
    const assignedByClip = new Map(pool.map(v => [v, 0]));
    const assignmentByCoder = new Map(assignments.map(a => [a.annotatorName, new Set(a.videos)]));

    for (const a of annotations) {
      if (!assignmentByCoder.get(a.annotatorName)?.has(a.videoFilename)) continue;
      doneByCoder.set(a.annotatorName, (doneByCoder.get(a.annotatorName) || 0) + 1);
      if (ratingsByClip.has(a.videoFilename)) ratingsByClip.set(a.videoFilename, ratingsByClip.get(a.videoFilename) + 1);
    }
    for (const a of assignments) {
      for (const v of a.videos) if (assignedByClip.has(v)) assignedByClip.set(v, assignedByClip.get(v) + 1);
    }

    for (const name of names) {
      coders[name] = coders[name] || { name };
      const mine = attempts.filter(a => a.annotatorName === name);
      coders[name][taskType] = {
        training: summarizeTraining(mine),
        assigned: assignmentByCoder.get(name)?.size || 0,
        completed: doneByCoder.get(name) || 0
      };
    }

    const target = config.assignment.ratingsPerClip;
    const ratingCounts = [...ratingsByClip.values()];
    const histogram = {};
    ratingCounts.forEach(c => { histogram[c] = (histogram[c] || 0) + 1; });
    coverage[taskType] = {
      clips: pool.length,
      ratingsPerClip: target,
      targetRatings: pool.length * target,
      assignedSlots: [...assignedByClip.values()].reduce((s, c) => s + c, 0),
      completedRatings: ratingCounts.reduce((s, c) => s + Math.min(c, target), 0),
      clipsFullyRated: ratingCounts.filter(c => c >= target).length,
      clipsByRatingCount: histogram
    };
  }

  // Fill in modalities a coder never started
  const blank = { training: summarizeTraining([]), assigned: 0, completed: 0 };
  const coderList = Object.values(coders).map(c => ({ do: blank, see: blank, ...c }))
    .sort((a, b) => a.name.localeCompare(b.name));

  res.json({
    success: true,
    config: { training: config.training, assignment: config.assignment },
    coders: coderList,
    coverage,
    goldStandardWarnings: goldStandards.warnings
  });
}));

app.get('/api/admin/export', requireAdmin, asyncRoute(async (req, res) => {
  const taskType = getTaskType(req);
  const filter = { taskType: taskType || undefined };
  if (req.query.training === 'true') filter.isTraining = true;
  if (req.query.training === 'false') filter.isTraining = false;
  const annotations = await store.findAnnotations(filter);
  sendAnnotationsCSV(res, annotations, `all_annotations_${taskType || 'all'}_${timestamp()}.csv`);
}));

// ============================================================================
// ROUTES - Gold standards editor (admin)
// ============================================================================

const GOLD_PATH = path.join(__dirname, 'gold_standards.csv');
const GOLD_EDITABLE = ['primary_activity', 'anyone_interacting', 'type', 'pa_reasoning', 'ai_reasoning', 'negative_of', 'disabled'];
// 'candidate' (or any other type) is kept in the CSV but ignored by training, so it works as a shortlist
const GOLD_TYPES = ['candidate', 'gold', 'example', 'confusing', 'na'];
// disabled = 'yes' turns a gold/example row off without losing its type
const GOLD_DISABLED = ['', 'yes'];

const isUsedGoldRow = (row) => ['gold', 'example'].includes((row.type || '').trim().toLowerCase()) &&
  (row.disabled || '').trim().toLowerCase() !== 'yes';

// What would make training skip a row (the same checks as lib/gold.js)
function goldRowProblems(row) {
  const taskType = (row.modality || '').trim().toLowerCase();
  const label = (row.primary_activity || '').trim().toLowerCase();
  const problems = [];
  if (!TASK_TYPES.includes(taskType)) {
    problems.push(`task is "${row.modality || ''}"; choose Doing or Seeing`);
  } else if (!config.activities[taskType].includes(label)) {
    problems.push(`"${row.primary_activity}" is not in the ${taskType} activity list`);
  }
  if (!['yes', 'no'].includes((row.anyone_interacting || '').trim().toLowerCase())) {
    problems.push('anyone_interacting must be yes or no');
  }
  if (!config.testMode && !inTrainingDirs(row.video_filename)) {
    problems.push(`the video file is not in ${config.videos.trainingDirs.join(', ')}/`);
  }
  return problems;
}

const goldRowJSON = (header, row, i) =>
  ({ index: i, csvRow: i + 2, ...Object.fromEntries(header.map(h => [h, row[h] ?? ''])), problems: goldRowProblems(row) });

// Every row of gold_standards.csv, with what's wrong with it (if anything)
app.get('/api/admin/gold', requireAdmin, asyncRoute(async (req, res) => {
  const { header, rows } = await readCSVWithHeader(GOLD_PATH);
  res.json({
    success: true,
    header,
    activities: { do: config.activities.do, see: config.activities.see },
    types: GOLD_TYPES,
    rows: rows.map((row, i) => goldRowJSON(header, row, i))
  });
}));

// Update one row's label/answer/reasoning; the previous file is kept in backups/
app.post('/api/admin/gold/:index', requireAdmin, asyncRoute(async (req, res) => {
  const { header, rows } = await readCSVWithHeader(GOLD_PATH);
  const i = parseInt(req.params.index, 10);
  const row = rows[i];
  if (!row) return res.status(404).json({ error: 'No such row' });
  // Guard against the file having changed since the page loaded
  if (req.body.video_filename !== row.video_filename) {
    return res.status(409).json({ error: 'gold_standards.csv changed since this page loaded; reload the page' });
  }

  const updates = {};
  for (const field of GOLD_EDITABLE) {
    if (req.body[field] !== undefined) updates[field] = String(req.body[field]).trim();
  }
  if (req.body.modality !== undefined) {
    updates.modality = String(req.body.modality).trim();
    if (!TASK_TYPES.includes(updates.modality)) return res.status(400).json({ error: "task must be 'do' or 'see'" });
    const duplicate = rows.findIndex((r, j) => j !== i && r.video_filename === row.video_filename &&
      (r.modality || '').trim().toLowerCase() === updates.modality);
    if (duplicate >= 0) {
      return res.status(409).json({ error: `This clip already has a ${updates.modality} row (row ${duplicate + 2}); edit that one` });
    }
  }
  const taskType = (updates.modality ?? row.modality ?? '').trim().toLowerCase();
  if (updates.primary_activity !== undefined && ['do', 'see'].includes(taskType) &&
      !config.activities[taskType].includes(updates.primary_activity)) {
    return res.status(400).json({ error: `"${updates.primary_activity}" is not a ${taskType} activity` });
  }
  if (updates.anyone_interacting !== undefined && !['yes', 'no'].includes(updates.anyone_interacting)) {
    return res.status(400).json({ error: 'anyone_interacting must be yes or no' });
  }
  if (updates.type !== undefined && !GOLD_TYPES.includes(updates.type)) {
    return res.status(400).json({ error: `type must be one of ${GOLD_TYPES.join(', ')}` });
  }
  if (updates.negative_of && ['do', 'see'].includes(taskType) && !config.activities[taskType].includes(updates.negative_of)) {
    return res.status(400).json({ error: `"${updates.negative_of}" is not a ${taskType} activity` });
  }
  if (updates.disabled !== undefined && !GOLD_DISABLED.includes(updates.disabled)) {
    return res.status(400).json({ error: "disabled must be 'yes' or blank" });
  }
  // A main clip in training would be graded in practice and then annotated again in the main task
  if (['gold', 'example'].includes(updates.type) && (await getMainVideoPool()).includes(row.video_filename)) {
    return res.status(400).json({ error: 'Main-task clips can only be candidate or confusing; as gold or example, coders would see the answer before annotating it' });
  }
  for (const field of ['negative_of', 'disabled']) {
    if (updates[field] !== undefined && !header.includes(field)) header.push(field);
  }

  const updated = { ...row, ...updates };
  // Don't switch on (or move to the other task) a row that training would skip
  const problems = goldRowProblems(updated);
  const switchedOn = !isUsedGoldRow(row) || updated.modality !== row.modality;
  if (isUsedGoldRow(updated) && switchedOn && problems.length) {
    return res.status(400).json({ error: `Row ${i + 2} can't be used yet: ${problems.join('; ')}` });
  }

  rows[i] = updated;
  writeCSVWithBackup(GOLD_PATH, header, rows);
  await reloadGoldStandards();
  console.log(`Gold standards row ${i + 2} updated by ${req.session.user?.username || 'admin'}:`, Object.keys(updates).join(', '));
  res.json({ success: true, row: goldRowJSON(header, updated, i), warnings: goldStandards.warnings });
}));

app.get('/api/admin/gold.csv', requireAdmin, (req, res) => {
  res.download(GOLD_PATH, 'gold_standards.csv');
});

// ============================================================================
// ROUTES - Annotation gallery (admin)
// ============================================================================

// Built by scripts/build_gallery_data.py:
//   data/original_pass.json    answers from the original doing/seeing pass and the model validation pass
//   data/model_detections.tsv  every clip the model labeled (activity, location, description)
const ORIGINAL_PASS_PATH = path.join(__dirname, 'data', 'original_pass.json');
const DETECTIONS_PATH = path.join(__dirname, 'data', 'model_detections.tsv');
let originalPass = { mtime: 0, data: null };
let detections = { mtime: 0, index: null };

function loadOriginalPass() {
  if (!fs.existsSync(ORIGINAL_PASS_PATH)) return null;
  const mtime = fs.statSync(ORIGINAL_PASS_PATH).mtimeMs;
  if (mtime !== originalPass.mtime) {
    const data = JSON.parse(fs.readFileSync(ORIGINAL_PASS_PATH, 'utf8'));
    // Earlier answers per old clip name, tagged with the pass they came from
    data.answers = { do: new Map(), see: new Map() };
    for (const taskType of TASK_TYPES) {
      const add = (clip, pass) => {
        const list = data.answers[taskType].get(clip.key) || [];
        list.push(...clip.votes.map(v => ({ ...v, pass })));
        data.answers[taskType].set(clip.key, list);
      };
      data.original[taskType].forEach(c => add(c, 'original'));
      data.validation.forEach(c => add(c, 'validation'));
    }
    originalPass = { mtime, data };
  }
  return originalPass.data;
}

// The TSV is ~100 MB, so only the short columns are kept in memory; a clip's description is read
// from the file (by byte offset) when the clip is shown
function loadDetections() {
  if (!fs.existsSync(DETECTIONS_PATH)) return null;
  const mtime = fs.statSync(DETECTIONS_PATH).mtimeMs;
  if (mtime === detections.mtime) return detections.index;

  const buf = fs.readFileSync(DETECTIONS_PATH);
  const rows = [];
  let start = buf.indexOf(10) + 1; // skip the header
  while (start < buf.length) {
    let end = buf.indexOf(10, start);
    if (end < 0) end = buf.length;
    const tabs = [];
    for (let i = start; i < end && tabs.length < 4; i++) if (buf[i] === 9) tabs.push(i);
    if (tabs.length === 4) {
      rows.push({
        key: buf.toString('utf8', start, tabs[0]),
        video: buf.toString('utf8', tabs[0] + 1, tabs[1]),
        activity: buf.toString('utf8', tabs[1] + 1, tabs[2]),
        location: buf.toString('utf8', tabs[2] + 1, tabs[3]),
        descStart: tabs[3] + 1,
        descEnd: end
      });
    }
    start = end + 1;
  }
  const byKey = new Map(rows.map(r => [r.key, r]));
  const byVideo = new Map(rows.filter(r => r.video).map(r => [path.basename(r.video), r]));
  detections = { mtime, index: { rows, byKey, byVideo } };
  console.log(`✓ Loaded ${rows.length} model detections from ${DETECTIONS_PATH}`);
  return detections.index;
}

function readDescription(row) {
  const fd = fs.openSync(DETECTIONS_PATH, 'r');
  try {
    const buf = Buffer.alloc(row.descEnd - row.descStart);
    fs.readSync(fd, buf, 0, buf.length, row.descStart);
    return buf.toString('utf8');
  } finally {
    fs.closeSync(fd);
  }
}

// Absolute path of a model-detected clip in the cluster's 2026.1 clip folder
const detectionPath = (row) => {
  const root = loadOriginalPass()?.inputs.source;
  return root && row?.video ? path.join(root, row.video) : null;
};

const inTrainingDirs = (filename) =>
  !!filename && config.videos.trainingDirs.some(dir => fs.existsSync(path.join(dir, filename)));

// gold_standards.csv rows per clip (new and old name), so the gallery can mark clips already in it
async function goldRowsByClip() {
  const { rows } = await readCSVWithHeader(GOLD_PATH);
  const byClip = new Map();
  rows.forEach((row, i) => {
    for (const name of new Set([row.video_filename, row.old_video_filename].filter(Boolean))) {
      if (!byClip.has(name)) byClip.set(name, []);
      byClip.get(name).push({
        csvRow: i + 2, videoFilename: row.video_filename, type: row.type, modality: row.modality.trim().toLowerCase(),
        primaryActivity: row.primary_activity, negativeOf: row.negative_of || '', disabled: row.disabled === 'yes'
      });
    }
  });
  return byClip;
}

const answerLabels = (a) => [a.primaryActivity, ...(a.otherActivities || [])];

function detectionSummary(row) {
  return row ? { activity: row.activity, location: row.location, description: readDescription(row) } : null;
}

// How well the earlier passes back up the model's label: the mean of description accuracy
// (1-5 scaled to 0-1) and the share of answers that mention the model's activity. null = never checked.
function validationScore(row, answers, activityList) {
  if (!answers.length) return null;
  const modelLabels = asCurrent(row.activity, activityList);
  const sameAs = (label) => label === row.activity || asCurrent(label, activityList).some(a => modelLabels.includes(a));
  const confirmed = answers.filter(a => [a.primaryActivity, ...(a.otherActivities || [])].some(sameAs)).length;
  const parts = [confirmed / answers.length];
  const ratings = answers.map(a => a.accuracy).filter(x => x != null);
  if (ratings.length) parts.push((ratings.reduce((s, x) => s + x, 0) / ratings.length - 1) / 4);
  return parts.reduce((s, x) => s + x, 0) / parts.length;
}

// Deterministic shuffle, so paging through a filtered set with the same seed doesn't repeat clips
function seededShuffle(items, seed) {
  let h = 1779033703 ^ seed.length;
  for (const ch of seed) h = Math.imul(h ^ ch.charCodeAt(0), 3432918353), h = (h << 13) | (h >>> 19);
  const rand = () => {
    h = Math.imul(h ^ (h >>> 16), 2246822507); h = Math.imul(h ^ (h >>> 13), 3266489909);
    return ((h ^= h >>> 16) >>> 0) / 4294967296;
  };
  const a = [...items];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// Gallery data. Sources:
//   current  the main-task clip pool, every clip whether or not it has been annotated yet, with the
//            answers saved in this app. All clips are returned; the page filters them.
//   model    every clip the model labeled, with any answers from the original pass and the model
//            validation pass. Filtered and paged here, since there are ~300k.
app.get('/api/admin/gallery', requireAdmin, requireTaskType, asyncRoute(async (req, res) => {
  const { taskType } = req;
  const activityList = config.activities[taskType];
  const gold = await goldRowsByClip();
  const goldFor = (...names) => [...new Map([...new Set(names.filter(Boolean))]
    .flatMap(n => gold.get(n) || []).map(g => [g.csvRow, g])).values()];
  const index = loadDetections();
  const earlier = loadOriginalPass();

  if (req.query.source !== 'model') {
    const latest = new Map();
    for (const a of await store.findAnnotations({ taskType, isTraining: false })) {
      const key = `${a.videoFilename}|${a.annotatorName}`;
      const prev = latest.get(key);
      if (!prev || new Date(a.updatedAt) > new Date(prev.updatedAt)) latest.set(key, a);
    }
    const votes = new Map();
    for (const a of latest.values()) {
      if (!votes.has(a.videoFilename)) votes.set(a.videoFilename, []);
      votes.get(a.videoFilename).push({
        annotatorName: a.annotatorName,
        primaryActivity: a.primaryActivity,
        primaryActivityOther: a.primaryActivityOther || '',
        anyoneInteracting: a.anyoneInteracting,
        updatedAt: a.updatedAt
      });
    }
    const clips = (await getMainVideoPool()).map(videoFilename => {
      const row = index?.byVideo.get(videoFilename);
      return {
        key: videoFilename,
        videoFilename,
        src: `/videos/${encodeURIComponent(videoFilename)}`,
        votes: votes.get(videoFilename) || [],
        model: detectionSummary(row),
        earlier: row ? earlier?.answers[taskType].get(row.key) || [] : [],
        gold: goldFor(videoFilename)
      };
    });
    const labels = clips.flatMap(c => [...c.votes, ...c.earlier].flatMap(answerLabels).concat(c.model?.activity));
    return res.json({ success: true, source: 'current', taskType, activities: config.activities, total: clips.length,
      clips, labelMap: labelMap(labels, activityList) });
  }

  if (!index) return res.status(404).json({ error: 'data/model_detections.tsv not found; run scripts/build_gallery_data.py' });

  // Filters: focus (a current activity) + relation, which earlier passes, model label, gold, search
  const focus = activityList.includes(req.query.focus) ? req.query.focus : '';
  const relation = req.query.relation || 'detected';
  const checked = req.query.checked || 'all';
  const search = String(req.query.search || '').trim().toLowerCase();
  const mapped = new Map();
  const toCurrent = (label) => {
    if (!mapped.has(label)) mapped.set(label, asCurrent(label, activityList));
    return mapped.get(label);
  };
  const answersFor = (row) => earlier?.answers[taskType].get(row.key) || [];

  let rows = index.rows.filter(row => {
    if (req.query.requireVideo !== 'false' && !row.video) return false;
    if (req.query.modelActivity && row.activity !== req.query.modelActivity) return false;
    if (search && !row.key.toLowerCase().includes(search) && !row.video.toLowerCase().includes(search)) return false;
    const answers = answersFor(row);
    if (checked === 'checked' && !answers.length) return false;
    if (checked === 'original' && !answers.some(a => a.pass === 'original')) return false;
    if (checked === 'validation' && !answers.some(a => a.pass === 'validation')) return false;
    if (checked === 'unchecked' && answers.length) return false;
    if (req.query.hideGold === 'true' &&
        goldFor(row.key, path.basename(row.video)).some(g => g.modality === taskType)) return false;
    if (!focus) return true;

    const modelSaysFocus = toCurrent(row.activity).includes(focus);
    const forFocus = answers.filter(a => toCurrent(a.primaryActivity).includes(focus)).length;
    const mentionsFocus = answers.some(a => [a.primaryActivity, ...(a.otherActivities || [])]
      .some(l => toCurrent(l).includes(focus)));
    if (relation === 'detected') return modelSaysFocus;
    // The model's detection is a candidate negative: coders mostly didn't see the activity
    if (relation === 'model-only') return modelSaysFocus && answers.length > 0 && forFocus <= answers.length / 2;
    // Coders saw the activity where the model labeled something else
    if (relation === 'coders-only') return !modelSaysFocus && mentionsFocus;
    return true;
  });

  if (req.query.sort === 'validation') {
    const scored = rows.map(row => [validationScore(row, answersFor(row), activityList), row]);
    scored.sort((a, b) => (a[0] ?? Infinity) - (b[0] ?? Infinity) || a[1].key.localeCompare(b[1].key));
    rows = scored.map(([, row]) => row);
  } else if (req.query.sort === 'name') {
    rows.sort((a, b) => a.key.localeCompare(b.key));
  } else {
    rows = seededShuffle(rows, String(req.query.seed || ''));
  }

  const offset = Math.max(0, parseInt(req.query.offset, 10) || 0);
  const limit = Math.min(100, Math.max(1, parseInt(req.query.limit, 10) || 24));
  const clips = rows.slice(offset, offset + limit).map(row => {
    const videoFilename = path.basename(row.video);
    const answers = answersFor(row);
    return {
      key: row.key,
      videoFilename,
      oldVideoFilename: row.key,
      src: inTrainingDirs(videoFilename) ? `/training-videos/${encodeURIComponent(videoFilename)}`
        : row.video ? `/api/admin/gallery/video?key=${encodeURIComponent(row.key)}` : null,
      model: detectionSummary(row),
      votes: [],
      earlier: answers,
      validationScore: validationScore(row, answers, activityList),
      gold: goldFor(row.key, videoFilename)
    };
  });
  const labels = clips.flatMap(c => c.earlier.flatMap(answerLabels).concat(c.model.activity));
  res.json({
    success: true, source: 'model', taskType, activities: config.activities, total: rows.length, offset, clips,
    modelActivities: [...new Set(index.rows.map(r => r.activity))].filter(Boolean).sort(),
    labelMap: labelMap(labels, activityList)
  });
}));

// Plays a model-detected clip from the cluster's 2026.1 clip folder (only clips listed in the TSV)
app.get('/api/admin/gallery/video', requireAdmin, (req, res) => {
  const file = detectionPath(loadDetections()?.byKey.get(String(req.query.key || '')));
  if (!file || !fs.existsSync(file)) return res.status(404).end();
  res.sendFile(file);
});

// Add a clip to gold_standards.csv. Model-detected clips are first copied into the training folder
// under their 2026.1 name. The previous CSV is kept in backups/.
app.post('/api/admin/gold', requireAdmin, asyncRoute(async (req, res) => {
  const body = req.body || {};
  const modality = String(body.modality || '').trim();
  const fields = Object.fromEntries(GOLD_EDITABLE.map(f => [f, String(body[f] ?? '').trim()]));

  if (!TASK_TYPES.includes(modality)) return res.status(400).json({ error: "modality must be 'do' or 'see'" });
  if (!config.activities[modality].includes(fields.primary_activity)) {
    return res.status(400).json({ error: `"${fields.primary_activity}" is not a ${modality} activity` });
  }
  if (!['yes', 'no'].includes(fields.anyone_interacting)) {
    return res.status(400).json({ error: 'anyone_interacting must be yes or no' });
  }
  if (!GOLD_TYPES.includes(fields.type)) {
    return res.status(400).json({ error: `type must be one of ${GOLD_TYPES.join(', ')}` });
  }
  if (fields.negative_of && !config.activities[modality].includes(fields.negative_of)) {
    return res.status(400).json({ error: `"${fields.negative_of}" is not a ${modality} activity` });
  }
  if (!GOLD_DISABLED.includes(fields.disabled)) {
    return res.status(400).json({ error: "disabled must be 'yes' or blank" });
  }

  let videoFilename, oldVideoFilename = '';
  if (body.source === 'model') {
    const row = loadDetections()?.byKey.get(String(body.key || ''));
    const file = detectionPath(row);
    if (!file || !fs.existsSync(file)) return res.status(400).json({ error: 'This clip has no 2026.1 version to copy' });
    videoFilename = path.basename(file);
    oldVideoFilename = row.key;
    if (!inTrainingDirs(videoFilename)) {
      fs.copyFileSync(file, path.join(config.videos.trainingDirs[0], videoFilename));
      fs.chmodSync(path.join(config.videos.trainingDirs[0], videoFilename), 0o644);
      console.log(`Copied ${file} into ${config.videos.trainingDirs[0]}`);
    }
  } else {
    videoFilename = String(body.video_filename || '');
    if (!inTrainingDirs(videoFilename) && !config.testMode) {
      return res.status(400).json({ error: `${videoFilename} is not in the training video folder` });
    }
    // A main clip in training would be graded in practice and then annotated again in the main task
    if (['gold', 'example'].includes(fields.type) && (await getMainVideoPool()).includes(videoFilename)) {
      return res.status(400).json({ error: 'Main-task clips can only be added as candidate or confusing; as gold or example, coders would see the answer before annotating it' });
    }
  }

  const { header, rows } = await readCSVWithHeader(GOLD_PATH);
  const duplicate = rows.findIndex(r => r.video_filename === videoFilename && r.modality.trim().toLowerCase() === modality);
  if (duplicate >= 0) {
    return res.status(409).json({ error: `Already in gold_standards.csv (row ${duplicate + 2}); edit it in the gold standards editor` });
  }
  for (const h of ['video_filename', 'old_video_filename', 'type', 'modality', ...GOLD_EDITABLE]) {
    if (!header.includes(h)) header.push(h);
  }

  rows.push({ ...fields, video_filename: videoFilename, old_video_filename: oldVideoFilename, modality });
  writeCSVWithBackup(GOLD_PATH, header, rows);
  await reloadGoldStandards();
  console.log(`Gold standards row ${rows.length + 1} added by ${req.session.user?.username || 'admin'}: ` +
    `${videoFilename} (${fields.type}, ${modality}, ${fields.primary_activity}${fields.negative_of ? `, not ${fields.negative_of}` : ''})`);
  res.json({ success: true, csvRow: rows.length + 1, videoFilename, warnings: goldStandards.warnings });
}));

// ============================================================================
// START SERVER
// ============================================================================

async function reloadGoldStandards() {
  goldStandards = await loadGoldStandards({
    csvPath: GOLD_PATH,
    activities: config.activities,
    trainingDirs: config.videos.trainingDirs,
    requireFiles: !config.testMode
  });
}

async function start() {
  const mongoUri = process.env.MONGODB_URI || 'mongodb://localhost:27017/activity_annotations';
  let sessionStore;
  if (config.testMode) {
    store = createMemoryStore();
    console.log('✓ TEST MODE: in-memory storage (data is lost on restart), placeholder videos, auth disabled');
  } else {
    store = await createMongoStore(mongoUri);
    sessionStore = require('connect-mongo').create({ mongoUrl: mongoUri });
  }
  if (config.devMode) console.log('✓ DEV MODE: skip controls enabled, training gates relaxed');
  if (usingAccounts) {
    const count = readUsers(config.usersPath).length;
    console.log(`✓ Accounts: ${count} user(s) in ${config.usersPath}`);
    if (count === 0) console.warn('⚠ No accounts yet; add one with: npm run users -- add <username> --admin');
    if (!process.env.SESSION_SECRET && !config.testMode) console.warn('⚠ SESSION_SECRET is not set; set it in .env');
  } else if (!config.testMode) {
    console.warn('⚠ auth.mode is "none": anyone can annotate under any ID and open the admin page');
  }

  sessionMiddleware = session({
    secret: process.env.SESSION_SECRET || 'secret-key-temp',
    resave: false,
    saveUninitialized: false,
    rolling: true,
    store: sessionStore,
    cookie: {
      maxAge: 1000 * 60 * 60 * 8,
      httpOnly: true,
      sameSite: 'lax',
      secure: server instanceof require('https').Server
    }
  });

  await reloadGoldStandards();

  server.listen(PORT, '0.0.0.0', () => {
    const protocol = server instanceof require('https').Server ? 'https' : 'http';
    console.log(`✓ Activity annotations at ${protocol}://localhost:${PORT}/experiment/activities.html`);
    console.log(`✓ Admin dashboard at ${protocol}://localhost:${PORT}/experiment/admin.html`);
  });
}

start().catch(err => {
  console.error('✗ Failed to start server:', err);
  process.exit(1);
});

module.exports = app;
