#!/usr/bin/env node
// Manage coder accounts in users.json
//
//   npm run users -- add <username> [--admin] [--password <pw>]   add a coder, or reset their password
//   npm run users -- remove <username>
//   npm run users -- list
//
// Without --password a random password is generated and printed once; it is not stored anywhere.

const crypto = require('crypto');
const path = require('path');
const { USERNAME_PATTERN, hashPassword, readUsers, writeUsers } = require('../lib/users');

const usersPath = process.env.USERS_PATH || path.join(__dirname, '..', 'users.json');
const [command, username, ...rest] = process.argv.slice(2);

function option(name) {
  const i = rest.indexOf(`--${name}`);
  return i === -1 ? undefined : rest[i + 1];
}

function fail(message) {
  console.error(message);
  process.exit(1);
}

// Readable random password: 4 groups of 4 letters/digits, without look-alike characters
function generatePassword() {
  const alphabet = 'abcdefghjkmnpqrstuvwxyz23456789';
  const bytes = crypto.randomBytes(16);
  const chars = [...bytes].map(b => alphabet[b % alphabet.length]).join('');
  return chars.match(/.{4}/g).join('-');
}

const users = readUsers(usersPath);

if (command === 'add') {
  if (!username || !USERNAME_PATTERN.test(username)) {
    fail('Usage: npm run users -- add <username> [--admin] [--password <pw>]\n' +
      'Usernames may contain letters, digits and . _ @ - (max 64 characters).');
  }
  const password = option('password') || generatePassword();
  if (password.length < 8) fail('Passwords must be at least 8 characters.');

  const existing = users.find(u => u.username === username);
  const admin = rest.includes('--admin') || (existing ? existing.admin : false);
  const entry = { username, passwordHash: hashPassword(password), admin, createdAt: existing?.createdAt || new Date().toISOString() };
  writeUsers(usersPath, existing ? users.map(u => u.username === username ? entry : u) : [...users, entry]);

  console.log(`${existing ? 'Updated' : 'Added'} ${admin ? 'admin ' : ''}user "${username}"`);
  if (!option('password')) console.log(`Password: ${password}   (shown once - it is not stored)`);
} else if (command === 'remove') {
  if (!users.some(u => u.username === username)) fail(`No user "${username}"`);
  writeUsers(usersPath, users.filter(u => u.username !== username));
  console.log(`Removed user "${username}" (their annotations are kept)`);
} else if (command === 'list') {
  if (users.length === 0) console.log(`No users in ${usersPath}`);
  users.forEach(u => console.log(`${u.username}${u.admin ? '  (admin)' : ''}  created ${u.createdAt || '?'}`));
} else {
  fail('Usage: npm run users -- add <username> [--admin] [--password <pw>] | remove <username> | list');
}
