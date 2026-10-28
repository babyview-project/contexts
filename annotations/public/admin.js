// admin.js - training status per annotator ID and rating coverage per task

const $ = (sel) => document.querySelector(sel);
let data = null;

function escapeHTML(value) {
    return String(value ?? '').replace(/[&<>"']/g, c =>
        ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

const pct = (x) => x == null ? '–' : `${Math.round(x * 100)}%`;

function login() {
    const modal = $('#login-modal');
    const form = $('#login-form');
    modal.hidden = false;
    form.username.focus();
    return new Promise(resolve => {
        form.onsubmit = async (e) => {
            e.preventDefault();
            const res = await fetch('/api/auth/login', {
                method: 'POST',
                credentials: 'include',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username: form.username.value, password: form.password.value })
            });
            form.password.value = '';
            if (!res.ok) {
                $('#login-error').textContent = (await res.json().catch(() => ({}))).error || 'Login failed. Please try again.';
                $('#login-error').hidden = false;
                return;
            }
            modal.hidden = true;
            resolve();
        };
    });
}

async function load() {
    const res = await fetch('/api/admin/status', { credentials: 'include' });
    if (res.status === 401) {
        await login();
        return load();
    }
    // Logged in with a coder (non-admin) account
    if (res.status === 403) {
        await fetch('/api/auth/logout', { method: 'POST', credentials: 'include' });
        $('#login-form').username.value = '';
        $('#login-error').textContent = 'That account is not an admin. Log in with an admin account.';
        $('#login-error').hidden = false;
        await login();
        return load();
    }
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error || res.statusText);
    data = await res.json();
    render();
    $('#updated').textContent = 'Updated ' + new Date().toLocaleTimeString();
}

function trainingCell(t) {
    if (t.passed) return `<span class="status-pill done">✓ Passed</span> <span class="muted">(${t.attemptsUsed} attempt${t.attemptsUsed === 1 ? '' : 's'})</span>`;
    if (t.attemptsUsed === 0) return '<span class="muted">Not started</span>';
    if (t.attemptsRemaining === 0) return `<span class="status-pill" style="background: var(--bad-bg); color: var(--bad);">✗ Failed</span> <span class="muted">(${t.attemptsUsed}/${t.maxAttempts})</span>`;
    return `<span class="status-pill todo">In progress</span> <span class="muted">(${t.attemptsUsed}/${t.maxAttempts})</span>`;
}

function render() {
    const { coverage, coders, goldStandardWarnings, config } = data;

    $('#warnings').innerHTML = goldStandardWarnings.length ? `
        <div class="warning-box">
            <strong>${goldStandardWarnings.length} gold standard row(s) skipped</strong> · <a href="gold.html">review and fix them →</a>
            <ul>${goldStandardWarnings.map(w => `<li>${escapeHTML(w)}</li>`).join('')}</ul>
        </div>` : '';

    const perCoder = config.assignment.videosPerCoder
        ? `${config.assignment.videosPerCoder} clips per coder`
        : 'every coder gets all clips';
    $('#coverage').innerHTML = ['do', 'see'].map(taskType => {
        const c = coverage[taskType];
        const histogram = Object.entries(c.clipsByRatingCount)
            .sort((a, b) => a[0] - b[0])
            .map(([n, clips]) => `${clips} clip${clips === 1 ? '' : 's'} with ${n} rating${n === '1' ? '' : 's'}`)
            .join(' · ');
        return `
            <h3>${taskType === 'do' ? 'Doing' : 'Seeing'}</h3>
            <div class="stat-grid">
                <div class="stat"><div class="value">${c.clips}</div><div class="label">clips in pool</div></div>
                <div class="stat"><div class="value">${c.completedRatings} / ${c.targetRatings}</div><div class="label">ratings done (target ${c.ratingsPerClip}/clip)</div></div>
                <div class="stat"><div class="value">${c.clipsFullyRated}</div><div class="label">clips with ${c.ratingsPerClip}+ ratings</div></div>
                <div class="stat"><div class="value">${c.assignedSlots}</div><div class="label">clip slots assigned (${perCoder})</div></div>
            </div>
            <p class="muted" style="font-size: 13px;">${histogram || 'No clips in pool'}</p>`;
    }).join('');

    const filter = $('#filter').value.trim().toLowerCase();
    const rows = coders.filter(c => c.name.toLowerCase().includes(filter));
    $('#coders').innerHTML = rows.length ? rows.map(c => `
        <tr>
            <td><strong>${escapeHTML(c.name)}</strong></td>
            ${['do', 'see'].map(t => `
                <td>${trainingCell(c[t].training)}</td>
                <td class="num">${pct(c[t].training.bestScore)}</td>
                <td class="num">${c[t].assigned ? `${c[t].completed} / ${c[t].assigned}` : '–'}</td>`).join('')}
        </tr>`).join('') : '<tr><td colspan="7" class="muted">No coders yet</td></tr>';
}

$('#refresh-btn').addEventListener('click', () => load().catch(err => alert(err.message)));
$('#filter').addEventListener('input', () => data && render());
load().catch(err => {
    $('#app').innerHTML = `<div class="card center"><h2>Could not load status</h2><p class="error">${escapeHTML(err.message)}</p></div>`;
});
