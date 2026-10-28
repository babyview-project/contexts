// labels.js - maps activity labels from earlier passes onto the current activity lists
//
// The original doing/seeing pass and the model (all_contexts) used older label sets, e.g. "playing",
// "meal time", "watching tv". A label maps to every current activity it could stand for, so
// "playing" counts toward each "playing with/without ..." category.

const OLD_LABELS = {
  'meal time': ['eating/drinking'],
  'eating': ['eating/drinking'],
  'drinking': ['eating/drinking'],
  'nursing': ['eating/drinking'],
  'reading': ['reading time'],
  'watching tv': ['screen time'],
  'looking at device': ['screen time'],
  'exploring': ['looking around'],
  'sitting': ['looking around'],
  'standing': ['looking around'],
  'lying down': ['looking around'],
  'being held': ['being moved around'],
  'moving around': ['walking', 'crawling', 'being moved around'],
  'music time': ['playing with instrument', 'dancing'],
  'gardening': ['playing with nature'],
  'none': ['no activity']
};

// Current activities (from `activityList`) that `label` could stand for
function asCurrent(label, activityList) {
  const l = String(label || '').trim().toLowerCase();
  if (activityList.includes(l)) return [l];
  if (l === 'playing') return activityList.filter(a => a.startsWith('playing'));
  return (OLD_LABELS[l] || []).filter(a => activityList.includes(a));
}

// { label: [current activities] } for every label in `labels`, for the gallery page
function labelMap(labels, activityList) {
  return Object.fromEntries([...new Set(labels)].filter(Boolean).map(l => [l, asCurrent(l, activityList)]));
}

module.exports = { asCurrent, labelMap };
