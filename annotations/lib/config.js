// config.js - loads config.json + activities.json and resolves run modes

const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');

const args = process.argv.slice(2);
const flag = (name, envVar) => args.includes(`--${name}`) || process.env[envVar] === 'true';

// --test: no videos or MongoDB needed (in-memory storage, placeholder videos)
// --dev:  shows skip controls (examples -> practice -> main) and relaxes training gates
const testMode = flag('test', 'TEST_MODE');
const devMode = flag('dev', 'DEV_MODE');

function readJSON(file) {
  return JSON.parse(fs.readFileSync(file, 'utf8'));
}

const configPath = process.env.CONFIG_PATH || path.join(ROOT, 'config.json');
const activitiesPath = process.env.ACTIVITIES_PATH || path.join(ROOT, 'activities.json');

const config = readJSON(configPath);
const activities = readJSON(activitiesPath);

// auth.mode: 'accounts' (username/password from users.json) or 'none' (coders type an ID).
// Test mode uses 'none' unless AUTH_MODE is set explicitly, so the login flow can be tested too.
const AUTH_MODES = ['accounts', 'none'];
const authMode = process.env.AUTH_MODE || (testMode ? 'none' : (config.auth && config.auth.mode) || 'accounts');
if (!AUTH_MODES.includes(authMode)) {
  throw new Error(`auth.mode must be one of ${AUTH_MODES.join(', ')} (got '${authMode}')`);
}

const resolveDir = (dir) => path.isAbsolute(dir) ? dir : path.join(ROOT, dir);

const videos = {
  mainDir: resolveDir(config.videos.mainDir),
  mainListFile: config.videos.mainListFile ? resolveDir(config.videos.mainListFile) : null,
  trainingDirs: config.videos.trainingDirs.map(resolveDir)
};

// Every activity in a task list needs a definition for the info panel. Definitions are per task
// (definitions.do, definitions.see), since e.g. "walking" means something different when seeing.
for (const taskType of ['do', 'see']) {
  for (const label of activities[taskType]) {
    if (!activities.definitions[taskType]?.[label]) {
      console.warn(`⚠ activities.json: no definition for "${label}" (${taskType})`);
    }
  }
}

module.exports = {
  ROOT,
  testMode,
  devMode,
  authMode,
  usersPath: process.env.USERS_PATH || path.join(ROOT, 'users.json'),
  assignment: config.assignment,
  training: config.training,
  videos,
  test: config.test || { fakeVideoCount: 12 },
  activities
};
