// users.js - coder accounts stored in users.json. Only salted scrypt hashes are stored, never passwords.
//
// users.json: { "users": [ { "username", "passwordHash", "admin", "createdAt" } ] }
// passwordHash format: scrypt$<N>$<r>$<p>$<salt base64>$<hash base64>

const fs = require('fs');
const crypto = require('crypto');

const SCRYPT = { N: 16384, r: 8, p: 1, keylen: 64 };
const USERNAME_PATTERN = /^[A-Za-z0-9._@-]{1,64}$/;

function hashPassword(password) {
  const salt = crypto.randomBytes(16);
  const hash = crypto.scryptSync(password, salt, SCRYPT.keylen, { N: SCRYPT.N, r: SCRYPT.r, p: SCRYPT.p });
  return ['scrypt', SCRYPT.N, SCRYPT.r, SCRYPT.p, salt.toString('base64'), hash.toString('base64')].join('$');
}

function verifyPassword(password, stored) {
  const [scheme, N, r, p, salt, hash] = (stored || '').split('$');
  if (scheme !== 'scrypt' || !salt || !hash) return false;
  const expected = Buffer.from(hash, 'base64');
  const actual = crypto.scryptSync(password, Buffer.from(salt, 'base64'), expected.length,
    { N: +N, r: +r, p: +p });
  return crypto.timingSafeEqual(actual, expected);
}

// A hash that never matches, so unknown usernames take as long to reject as wrong passwords
const DUMMY_HASH = hashPassword(crypto.randomBytes(16).toString('hex'));

function readUsers(usersPath) {
  if (!fs.existsSync(usersPath)) return [];
  return JSON.parse(fs.readFileSync(usersPath, 'utf8')).users || [];
}

function writeUsers(usersPath, users) {
  fs.writeFileSync(usersPath, JSON.stringify({ users }, null, 2) + '\n', { mode: 0o600 });
  fs.chmodSync(usersPath, 0o600); // owner read/write only, also when the file already existed
}

// Read on every login so accounts can be added without restarting the server
function authenticate(usersPath, username, password) {
  const user = readUsers(usersPath).find(u => u.username === username);
  const ok = verifyPassword(String(password || ''), user ? user.passwordHash : DUMMY_HASH);
  return ok && user ? { username: user.username, admin: !!user.admin } : null;
}

// Simple in-memory brute-force protection: too many failures for a username or IP locks it briefly
function createLoginLimiter({ maxFailures = 10, windowMs = 15 * 60 * 1000 } = {}) {
  const failures = new Map();
  const recent = (key) => (failures.get(key) || []).filter(t => Date.now() - t < windowMs);
  return {
    isBlocked: (...keys) => keys.some(k => recent(k).length >= maxFailures),
    recordFailure: (...keys) => keys.forEach(k => failures.set(k, [...recent(k), Date.now()])),
    reset: (...keys) => keys.forEach(k => failures.delete(k))
  };
}

module.exports = {
  USERNAME_PATTERN,
  hashPassword,
  verifyPassword,
  readUsers,
  writeUsers,
  authenticate,
  createLoginLimiter
};
