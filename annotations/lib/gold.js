// gold.js - loads gold_standards.csv, validates labels, picks practice sets and grades them

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const csv = require('csv-parser');

const norm = (s) => (s || '').trim().toLowerCase();

function readCSV(csvPath) {
  return new Promise((resolve, reject) => {
    const rows = [];
    fs.createReadStream(csvPath)
      .on('error', reject)
      .pipe(csv())
      .on('data', row => rows.push(row))
      .on('end', () => resolve(rows))
      .on('error', reject);
  });
}

// Raw rows plus the header order, for the gold standards editor
function readCSVWithHeader(csvPath) {
  return new Promise((resolve, reject) => {
    const rows = [];
    let header = [];
    fs.createReadStream(csvPath)
      .on('error', reject)
      .pipe(csv())
      .on('headers', h => { header = h; })
      .on('data', row => rows.push(row))
      .on('end', () => resolve({ header, rows }))
      .on('error', reject);
  });
}

// Rewrites the CSV (columns in header order; stray extra columns are dropped) after copying the
// current file to backups/<name>.<timestamp>.csv
function writeCSVWithBackup(csvPath, header, rows) {
  const { createObjectCsvStringifier } = require('csv-writer');
  const backupDir = path.join(path.dirname(csvPath), 'backups');
  fs.mkdirSync(backupDir, { recursive: true });
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  fs.copyFileSync(csvPath, path.join(backupDir, `${path.basename(csvPath, '.csv')}.${stamp}.csv`));

  const stringifier = createObjectCsvStringifier({ header: header.map(id => ({ id, title: id })) });
  const clean = rows.map(r => Object.fromEntries(header.map(h => [h, r[h] ?? ''])));
  const tmp = csvPath + '.tmp';
  fs.writeFileSync(tmp, stringifier.getHeaderString() + stringifier.stringifyRecords(clean));
  fs.renameSync(tmp, csvPath);
}

function videoExists(filename, dirs) {
  return dirs.some(dir => fs.existsSync(path.join(dir, filename)));
}

// Returns { do: { examples, gold }, see: { examples, gold }, warnings }
async function loadGoldStandards({ csvPath, activities, trainingDirs, requireFiles }) {
  const result = { do: { examples: [], gold: [] }, see: { examples: [], gold: [] }, warnings: [] };
  const warn = (msg) => { result.warnings.push(msg); console.warn('⚠ ' + msg); };

  if (!fs.existsSync(csvPath)) {
    warn(`Gold standards CSV not found at ${csvPath}; training will not work`);
    return result;
  }

  const rows = await readCSV(csvPath);
  rows.forEach((row, i) => {
    const taskType = norm(row.modality) === 'see' ? 'see' : norm(row.modality) === 'do' ? 'do' : null;
    const type = norm(row.type);
    if (!taskType || !['example', 'gold'].includes(type)) return; // 'na', 'confusing', etc.
    if (norm(row.disabled) === 'yes') return; // turned off in the editor or gallery

    const item = {
      videoFilename: (row.video_filename || '').trim(),
      order: parseInt(row.order, 10),
      csvRow: i + 2,
      primaryActivity: norm(row.primary_activity),
      anyoneInteracting: norm(row.anyone_interacting),
      primaryActivityReasoning: (row.pa_reasoning || '').trim(),
      interactingReasoning: (row.ai_reasoning || '').trim()
    };
    const where = `gold_standards.csv row ${item.csvRow} (${type}, ${taskType}, ${item.videoFilename})`;

    if (!activities[taskType].includes(item.primaryActivity)) {
      warn(`${where}: "${item.primaryActivity}" is not in the ${taskType} activity list; skipped`);
      return;
    }
    if (!['yes', 'no'].includes(item.anyoneInteracting)) {
      warn(`${where}: anyone_interacting must be yes/no; skipped`);
      return;
    }
    if (requireFiles && !videoExists(item.videoFilename, trainingDirs)) {
      warn(`${where}: video file not found; skipped`);
      return;
    }
    result[taskType][type === 'example' ? 'examples' : 'gold'].push(item);
  });

  // Examples follow the 'order' column; rows without one keep CSV order at the end
  for (const taskType of ['do', 'see']) {
    result[taskType].examples.sort((a, b) =>
      (isNaN(a.order) ? Infinity : a.order) - (isNaN(b.order) ? Infinity : b.order) || a.csvRow - b.csvRow);
    console.log(`✓ ${taskType}: ${result[taskType].examples.length} examples, ${result[taskType].gold.length} gold standard clips`);
  }
  return result;
}

// Deterministic shuffle so a coder who reloads gets the same practice set for the same attempt
function seededShuffle(items, seed) {
  const keyed = items.map(item => ({
    item,
    key: crypto.createHash('sha1').update(seed + '|' + item.videoFilename).digest('hex')
  }));
  return keyed.sort((a, b) => a.key.localeCompare(b.key)).map(k => k.item);
}

// A set never has two clips with the same primary activity, so it is smaller than videosPerAttempt
// only if the pool has fewer activities. Attempts walk through the shuffled pool: each attempt takes
// clips no earlier attempt used first, and starts over once every clip has been used.
function practiceSet(goldPool, { annotatorName, taskType, attemptNumber, videosPerAttempt }) {
  if (goldPool.length === 0) return [];
  const shuffled = seededShuffle(goldPool, `${annotatorName}|${taskType}`);
  const n = Math.min(videosPerAttempt, new Set(shuffled.map(g => g.primaryActivity)).size);
  let used = new Set();
  let set = [];
  for (let attempt = 1; attempt <= attemptNumber; attempt++) {
    if (used.size === shuffled.length) used = new Set();
    const labels = new Set();
    set = [];
    for (const g of [...shuffled.filter(g => !used.has(g)), ...shuffled.filter(g => used.has(g))]) {
      if (set.length === n) break;
      if (labels.has(g.primaryActivity)) continue;
      labels.add(g.primaryActivity);
      set.push(g);
    }
    set.forEach(g => used.add(g));
  }
  return set;
}

// A clip is correct when both the primary activity and the interaction answer match
function grade(goldPool, responses) {
  const byFile = new Map(goldPool.map(g => [g.videoFilename, g]));
  const results = responses.map(r => {
    const gold = byFile.get(r.videoFilename);
    if (!gold) return null;
    const primaryCorrect = norm(r.primaryActivity) === gold.primaryActivity;
    const interactingCorrect = norm(r.anyoneInteracting) === gold.anyoneInteracting;
    return {
      videoFilename: r.videoFilename,
      response: {
        primaryActivity: r.primaryActivity,
        primaryActivityOther: r.primaryActivityOther || '',
        anyoneInteracting: r.anyoneInteracting
      },
      correctAnswer: {
        primaryActivity: gold.primaryActivity,
        anyoneInteracting: gold.anyoneInteracting,
        primaryActivityReasoning: gold.primaryActivityReasoning,
        interactingReasoning: gold.interactingReasoning
      },
      primaryCorrect,
      interactingCorrect,
      correct: primaryCorrect && interactingCorrect
    };
  });
  if (results.includes(null)) return null;
  const nCorrect = results.filter(r => r.correct).length;
  return { results, nCorrect, nTotal: results.length, score: results.length ? nCorrect / results.length : 0 };
}

module.exports = { loadGoldStandards, practiceSet, grade, readCSV, readCSVWithHeader, writeCSVWithBackup };
