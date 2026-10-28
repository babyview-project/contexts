// store.js - persistence for users, annotations, assignments and training attempts.
// MongoDB in normal runs; an in-memory store in test mode so nothing needs to be installed.

// Annotations are unique per (annotator, clip, task, training flag, attempt)
const annotationKey = (a) =>
  [a.annotatorName, a.videoFilename, a.taskType, !!a.isTraining, a.attemptNumber ?? ''].join('|');

function createMemoryStore() {
  const users = new Map();
  const annotations = new Map();
  const assignments = new Map();
  const attempts = new Map();

  const matches = (doc, filter) =>
    Object.entries(filter).every(([k, v]) => v === undefined || doc[k] === v);

  return {
    kind: 'memory',

    async upsertUser(name) {
      if (!users.has(name)) users.set(name, { name, createdAt: new Date() });
      return users.get(name);
    },
    async listUsers() {
      return [...users.values()];
    },

    async saveAnnotation(doc) {
      const key = annotationKey(doc);
      const saved = { ...annotations.get(key), ...doc, updatedAt: new Date() };
      annotations.set(key, saved);
      return saved;
    },
    async findAnnotations(filter) {
      return [...annotations.values()].filter(a => matches(a, filter));
    },

    async getAssignment(annotatorName, taskType) {
      return assignments.get(`${annotatorName}|${taskType}`) || null;
    },
    async createAssignment(doc) {
      const saved = { ...doc, createdAt: new Date() };
      assignments.set(`${doc.annotatorName}|${doc.taskType}`, saved);
      return saved;
    },
    async listAssignments(taskType) {
      return [...assignments.values()].filter(a => !taskType || a.taskType === taskType);
    },

    async saveTrainingAttempt(doc) {
      const saved = { ...doc, createdAt: new Date() };
      attempts.set(`${doc.annotatorName}|${doc.taskType}|${doc.attemptNumber}`, saved);
      return saved;
    },
    async listTrainingAttempts(filter) {
      return [...attempts.values()]
        .filter(a => matches(a, filter))
        .sort((a, b) => a.attemptNumber - b.attemptNumber);
    }
  };
}

async function createMongoStore(mongoUri) {
  const mongoose = require('mongoose');
  await mongoose.connect(mongoUri);
  console.log('✓ Connected to MongoDB');

  const User = mongoose.model('User', new mongoose.Schema({
    name: { type: String, required: true, unique: true },
    createdAt: { type: Date, default: Date.now }
  }));

  const AnnotationSchema = new mongoose.Schema({
    annotatorName: { type: String, required: true, index: true },
    videoFilename: { type: String, required: true, index: true },
    taskType: { type: String, enum: ['do', 'see'], default: 'do' },
    isTraining: { type: Boolean, default: false },
    attemptNumber: { type: Number, default: null },
    primaryActivity: { type: String, required: true },
    primaryActivityOther: String,
    anyoneInteracting: { type: String, required: true },
    updatedAt: { type: Date, default: Date.now }
  });
  AnnotationSchema.index({ annotatorName: 1, videoFilename: 1, taskType: 1, isTraining: 1, attemptNumber: 1 });
  const Annotation = mongoose.model('Annotation', AnnotationSchema);

  const AssignmentSchema = new mongoose.Schema({
    annotatorName: { type: String, required: true },
    taskType: { type: String, required: true },
    videos: [String],
    createdAt: { type: Date, default: Date.now }
  });
  AssignmentSchema.index({ annotatorName: 1, taskType: 1 }, { unique: true });
  const Assignment = mongoose.model('Assignment', AssignmentSchema);

  const TrainingAttemptSchema = new mongoose.Schema({
    annotatorName: { type: String, required: true, index: true },
    taskType: { type: String, required: true },
    attemptNumber: { type: Number, required: true },
    nCorrect: Number,
    nTotal: Number,
    score: Number,
    passed: Boolean,
    devMode: Boolean,
    results: mongoose.Schema.Types.Mixed,
    createdAt: { type: Date, default: Date.now }
  });
  TrainingAttemptSchema.index({ annotatorName: 1, taskType: 1, attemptNumber: 1 }, { unique: true });
  const TrainingAttempt = mongoose.model('TrainingAttempt', TrainingAttemptSchema);

  // Annotations from before taskType existed have no taskType and count as 'do'
  const taskFilter = (taskType) =>
    taskType === 'do' ? { taskType: { $ne: 'see' } } : taskType ? { taskType } : {};

  const clean = (filter) =>
    Object.fromEntries(Object.entries(filter).filter(([, v]) => v !== undefined));

  return {
    kind: 'mongo',
    sessionMongoUrl: mongoUri,

    async upsertUser(name) {
      return User.findOneAndUpdate({ name }, { $setOnInsert: { name } }, { upsert: true, new: true }).lean();
    },
    async listUsers() {
      return User.find().lean();
    },

    async saveAnnotation(doc) {
      const key = {
        annotatorName: doc.annotatorName,
        videoFilename: doc.videoFilename,
        taskType: doc.taskType,
        isTraining: !!doc.isTraining,
        attemptNumber: doc.attemptNumber ?? null
      };
      return Annotation.findOneAndUpdate(key, { ...doc, ...key, updatedAt: new Date() },
        { upsert: true, new: true, runValidators: true }).lean();
    },
    async findAnnotations({ taskType, isTraining, ...rest }) {
      const filter = { ...clean(rest), ...taskFilter(taskType) };
      // Older documents may be missing isTraining
      if (isTraining === false) filter.isTraining = { $ne: true };
      if (isTraining === true) filter.isTraining = true;
      return Annotation.find(filter).lean();
    },

    async getAssignment(annotatorName, taskType) {
      return Assignment.findOne({ annotatorName, taskType }).lean();
    },
    async createAssignment(doc) {
      return (await Assignment.create(doc)).toObject();
    },
    async listAssignments(taskType) {
      return Assignment.find(taskType ? { taskType } : {}).lean();
    },

    async saveTrainingAttempt(doc) {
      const key = { annotatorName: doc.annotatorName, taskType: doc.taskType, attemptNumber: doc.attemptNumber };
      return TrainingAttempt.findOneAndUpdate(key, doc, { upsert: true, new: true }).lean();
    },
    async listTrainingAttempts(filter) {
      return TrainingAttempt.find(clean(filter)).sort({ attemptNumber: 1 }).lean();
    }
  };
}

module.exports = { createMemoryStore, createMongoStore };
