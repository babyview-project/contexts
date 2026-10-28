// gold.js - review and fix gold standard / example clips (admin only)

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
let data = null;

function escapeHTML(value) {
    return String(value ?? '').replace(/[&<>"']/g, c =>
        ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function toast(message, isError = false) {
    const el = $('#toast');
    el.textContent = message;
    el.classList.toggle('error', isError);
    el.classList.add('show');
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => el.classList.remove('show'), isError ? 5000 : 2500);
}

function login(message) {
    const modal = $('#login-modal');
    const form = $('#login-form');
    if (message) {
        $('#login-error').textContent = message;
        $('#login-error').hidden = false;
        form.username.value = '';
    }
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
                $('#login-error').textContent = (await res.json().catch(() => ({}))).error || 'Login failed.';
                $('#login-error').hidden = false;
                return;
            }
            modal.hidden = true;
            resolve();
        };
    });
}

async function api(path, options = {}) {
    const res = await fetch('/api' + path, {
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        ...options
    });
    if (res.status === 401) {
        await login();
        return api(path, options);
    }
    if (res.status === 403) {
        await fetch('/api/auth/logout', { method: 'POST', credentials: 'include' });
        await login('That account is not an admin. Log in with an admin account.');
        return api(path, options);
    }
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.error || `Request failed (${res.status})`);
    return body;
}

async function load() {
    data = await api('/admin/gold');
    render();
}

const isUsedType = (type) => ['example', 'gold'].includes((type || '').trim().toLowerCase());
const isDisabled = (r) => (r.disabled || '').trim().toLowerCase() === 'yes';
const isUsed = (r) => isUsedType(r.type) && !isDisabled(r);

function render() {
    const show = $('input[name=show]:checked').value;
    const task = $('#task-filter').value;
    const type = $('#type-filter').value;

    const rows = data.rows.filter(r => {
        if (show === 'problems' && (r.problems.length === 0 || !isUsed(r))) return false;
        if (task && r.modality.trim().toLowerCase() !== task) return false;
        if (type === 'unused') return !isUsed(r);
        if (type === 'other' && isUsedType(r.type)) return false;
        if (type && type !== 'other' && r.type.trim().toLowerCase() !== type) return false;
        return true;
    });
    $('#count').textContent = `${rows.length} row${rows.length === 1 ? '' : 's'}`;

    $('#gold-list').innerHTML = rows.length ? rows.map(rowHTML).join('') :
        `<div class="card center"><p>${show === 'problems' ? 'Nothing needs fixing 🎉' : 'No rows match.'}</p></div>`;

    $$('.gold-item').forEach(bindRow);
}

const taskName = (t) => t === 'see' ? 'Seeing' : t === 'do' ? 'Doing' : '';
const activityOptions = (list, selected) =>
    list.map(o => `<option value="${escapeHTML(o)}" ${o === selected ? 'selected' : ''}>${escapeHTML(o)}</option>`).join('');

// Whether training uses the row, with buttons to switch it on or off. Turning a gold/example row off
// keeps its type (disabled = yes); other types are switched on by making them gold or example.
// The buttons use the task chosen in the row's Task field.
function statusHTML(r, task = r.modality.trim().toLowerCase()) {
    const forTask = taskName(task) ? ` for ${taskName(task).toLowerCase()}` : '';
    if (isUsed(r)) {
        return `<span class="use-pill on">✓ Used by the app as ${escapeHTML(r.type.trim())}${forTask}</span>
            <button type="button" class="btn btn-small" data-use='{"disabled":"yes"}'>Stop using</button>`;
    }
    if (isUsedType(r.type)) {
        return `<span class="use-pill off">Turned off · not used by the app</span>
            <button type="button" class="btn btn-small" data-use='{"disabled":""}'>Use again as ${escapeHTML(r.type.trim())}${forTask}</button>`;
    }
    return `<span class="use-pill off">Not used by the app (${escapeHTML(r.type || 'no type')})</span>
        <button type="button" class="btn btn-small" data-use='{"type":"gold","disabled":""}'>Use as gold${forTask}</button>
        <button type="button" class="btn btn-small" data-use='{"type":"example","disabled":""}'>Use as example${forTask}</button>`;
}

function rowHTML(r) {
    const taskType = r.modality.trim().toLowerCase();
    const options = data.activities[taskType] || [];
    const current = r.primary_activity.trim().toLowerCase();
    const src = `/training-videos/${encodeURIComponent(r.video_filename)}`;
    return `
        <div class="gold-item ${r.problems.length ? 'has-problem' : ''} ${isUsed(r) ? '' : 'unused'}" data-index="${r.index}">
            <div class="video-frame">
                <video data-src="${escapeHTML(src)}" controls preload="none" playsinline></video>
                <div class="video-placeholder" hidden><strong>Video could not be loaded</strong><span>${escapeHTML(r.video_filename)}</span></div>
            </div>
            <div>
                <div class="gold-meta">
                    <strong>Row ${r.csvRow}</strong> · ${escapeHTML(r.type || '—')} · ${taskName(taskType) || escapeHTML(r.modality || '—')}
                    ${r.order ? ` · order ${escapeHTML(r.order)}` : ''}<br>
                    ${escapeHTML(r.video_filename)}${r.old_video_filename && r.old_video_filename !== r.video_filename ? `<br>was ${escapeHTML(r.old_video_filename)}` : ''}
                </div>
                <div class="gold-status">${statusHTML(r)}</div>
                ${r.problems.map(p => `<div class="gold-problem">⚠ ${escapeHTML(p)}</div>`).join('')}
                <div class="gold-fields">
                    <label>Task
                        <select name="modality">
                            ${taskName(taskType) ? '' : `<option value="" selected>${escapeHTML(r.modality || '(none)')} — choose a task…</option>`}
                            ${['do', 'see'].map(t => `<option value="${t}" ${t === taskType ? 'selected' : ''}>${taskName(t)}</option>`).join('')}
                        </select>
                    </label>
                    <label>Type
                        <select name="type">
                            ${data.types.includes(r.type.trim()) ? '' : `<option value="" selected>${escapeHTML(r.type || '(none)')} — choose a type…</option>`}
                            ${data.types.map(t => `<option ${r.type.trim() === t ? 'selected' : ''}>${t}</option>`).join('')}
                        </select>
                    </label>
                    <label>Primary activity
                        <select name="primary_activity">
                            ${options.includes(current) ? '' : `<option value="" selected>${escapeHTML(r.primary_activity || '(none)')} — choose a new label…</option>`}
                            ${activityOptions(options, current)}
                        </select>
                    </label>
                    <label>Anyone interacting
                        <select name="anyone_interacting">
                            ${['yes', 'no'].map(v => `<option ${r.anyone_interacting.trim().toLowerCase() === v ? 'selected' : ''}>${v}</option>`).join('')}
                        </select>
                    </label>
                    <label class="full">Negative example of
                        <select name="negative_of">
                            <option value="">—</option>
                            ${activityOptions(options, r.negative_of || '')}
                        </select>
                    </label>
                    <label class="full">Primary activity reasoning
                        <textarea name="pa_reasoning">${escapeHTML(r.pa_reasoning)}</textarea>
                    </label>
                    <label class="full">Interaction reasoning
                        <textarea name="ai_reasoning">${escapeHTML(r.ai_reasoning)}</textarea>
                    </label>
                </div>
                <div class="gold-actions">
                    <button type="button" class="btn btn-primary" data-save>Save row ${r.csvRow}</button>
                    <span class="saved-note" hidden>✓ Saved</span>
                </div>
            </div>
        </div>`;
}

// Videos load when they scroll into view and are released when they scroll away (unless playing).
// Loading every row's video at once ties up the browser's few connections to the server, and the
// videos further down then never load.
const lazyVideos = new IntersectionObserver(entries => {
    for (const { target: video, isIntersecting } of entries) {
        if (isIntersecting && !video.getAttribute('src')) {
            video.src = video.dataset.src;
            video.preload = 'metadata';
        } else if (!isIntersecting && video.paused && video.getAttribute('src')) {
            video.removeAttribute('src');
            video.load();
        }
    }
}, { rootMargin: '400px' });

function bindRow(el) {
    const row = data.rows.find(r => r.index === +el.dataset.index);
    const video = $('video', el);
    video.addEventListener('error', () => {
        if (!video.getAttribute('src')) return;
        video.hidden = true;
        video.nextElementSibling.hidden = false;
    });
    lazyVideos.observe(video);

    const note = $('.saved-note', el);
    $$('select, textarea', el).forEach(input => input.addEventListener('input', () => { note.hidden = true; }));

    // Switching the task switches the activity lists (keeping the label if the other task has it too)
    const field = (name) => $(`[name=${name}]`, el);
    field('modality').addEventListener('change', () => {
        const list = data.activities[field('modality').value] || [];
        const label = field('primary_activity').value;
        field('primary_activity').innerHTML = (list.includes(label) ? '' : '<option value="" selected>Choose a label…</option>') +
            activityOptions(list, label);
        field('negative_of').innerHTML = '<option value="">—</option>' + activityOptions(list, field('negative_of').value);
        bindStatus(el, row);
    });
    bindStatus(el, row);

    $('[data-save]', el).addEventListener('click', () => {
        if (!field('modality').value) return toast('Choose Doing or Seeing first', true);
        if (!field('primary_activity').value) return toast('Choose a primary activity first', true);
        if (!field('type').value) return toast('Choose a type first', true);
        save(el, row, {}, 'Saved');
    });
}

function bindStatus(el, row) {
    $('.gold-status', el).innerHTML = statusHTML(row, $('[name=modality]', el).value);
    $$('[data-use]', el).forEach(btn => btn.addEventListener('click', () => {
        const changes = JSON.parse(btn.dataset.use);
        const turningOn = changes.disabled === '';
        if (turningOn && !$('[name=modality]', el).value) return toast('Choose Doing or Seeing in the Task field first', true);
        if (turningOn && !$('[name=primary_activity]', el).value) return toast('Choose a primary activity first', true);
        save(el, row, changes, turningOn ? 'Now used' : 'No longer used');
    }));
}

// Saves the whole row (so nothing typed is lost) plus `changes`, then redraws it from the server's copy
async function save(el, row, changes, verb) {
    const value = (name) => $(`[name=${name}]`, el).value;
    const fields = {
        modality: value('modality'),
        type: value('type'),
        primary_activity: value('primary_activity'),
        anyone_interacting: value('anyone_interacting'),
        negative_of: value('negative_of'),
        pa_reasoning: value('pa_reasoning'),
        ai_reasoning: value('ai_reasoning')
    };
    // Blank selects mean "not chosen yet": leave the CSV value as it is
    for (const key of ['modality', 'type', 'primary_activity']) if (!fields[key]) delete fields[key];
    $$('button', el).forEach(b => { b.disabled = true; });
    try {
        const result = await api(`/admin/gold/${row.index}`, {
            method: 'POST',
            body: JSON.stringify({ video_filename: row.video_filename, ...fields, ...changes })
        });
        Object.assign(row, result.row);
        const scrollY = window.scrollY;
        el.outerHTML = rowHTML(row);
        const fresh = $(`.gold-item[data-index="${row.index}"]`);
        bindRow(fresh);
        window.scrollTo(0, scrollY);
        $('.saved-note', fresh).hidden = false;
        const task = taskName(row.modality.trim().toLowerCase()).toLowerCase();
        toast(`${verb}: row ${row.csvRow}${isUsed(row) ? ` (${row.type.trim()}${task ? `, ${task}` : ''})` : ''}`);
    } catch (error) {
        toast(error.message, true);
        $$('button', el).forEach(b => { b.disabled = false; });
    }
}

$$('input[name=show]').forEach(r => r.addEventListener('change', render));
$('#task-filter').addEventListener('change', render);
$('#type-filter').addEventListener('change', render);
load().catch(err => {
    $('#gold-list').innerHTML = `<div class="card center"><p class="error">${escapeHTML(err.message)}</p></div>`;
});
