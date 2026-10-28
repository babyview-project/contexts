// gallery.js - browse annotated clips to find new gold standard / negative examples (admin only)
//
// Two sources:
//   current  the main-task clip pool with the answers saved in this app (filtered here)
//   model    every clip the model labeled, with answers from earlier passes (filtered and paged by the server)

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const PAGE_SIZE = 24;

const state = {
    source: 'current', task: 'do', focus: '', relCurrent: 'near', relModel: 'detected', checked: 'all',
    modelActivity: '', agreement: '', interaction: '', minVotes: '0', sortCurrent: 'agreement',
    sortModel: 'random', search: '', hideGold: false, seed: Math.random().toString(36).slice(2, 8)
};
const CONTROLS = {
    focus: '#focus', relCurrent: '#relation-current', relModel: '#relation-model', checked: '#checked',
    modelActivity: '#model-activity', agreement: '#agreement', interaction: '#interaction',
    minVotes: '#min-votes', sortCurrent: '#sort-current', sortModel: '#sort-model', search: '#search'
};
let data = null;       // last response from /api/admin/gallery
let clips = [];        // clips currently listed (filtered + sorted)
let shown = 0;
let modelActivities = [];
let requestId = 0;

function escapeHTML(value) {
    return String(value ?? '').replace(/[&<>"']/g, c =>
        ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

const pct = (x) => `${Math.round(x * 100)}%`;
const level = (x) => x <= 0.5 ? 'low' : x < 1 ? 'mid' : 'high';
const taskName = (t) => t === 'see' ? 'Seeing' : 'Doing';

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

// ---------------------------------------------------------------------------
// State <-> URL and controls
// ---------------------------------------------------------------------------

function readURL() {
    const params = new URLSearchParams(location.search);
    for (const key of Object.keys(state)) {
        if (!params.has(key)) continue;
        state[key] = typeof state[key] === 'boolean' ? params.get(key) === 'true' : params.get(key);
    }
    if (!['current', 'model'].includes(state.source)) state.source = 'current';
    if (!['do', 'see'].includes(state.task)) state.task = 'do';
}

function writeURL() {
    const params = new URLSearchParams();
    for (const [key, value] of Object.entries(state)) if (value !== '' && value !== false) params.set(key, value);
    history.replaceState(null, '', `${location.pathname}?${params}`);
}

function syncControls() {
    $$('#source button').forEach(b => b.classList.toggle('active', b.dataset.value === state.source));
    $$('#task button').forEach(b => b.classList.toggle('active', b.dataset.value === state.task));
    $$('[data-for]').forEach(el => { el.hidden = el.dataset.for !== state.source; });
    $('#hide-gold').checked = state.hideGold;
    $('#search').placeholder = state.source === 'model' ? 'Filename…' : 'Filename or coder…';

    const activities = data?.activities[state.task] || [];
    if (state.focus && activities.length && !activities.includes(state.focus)) state.focus = '';
    $('#focus').innerHTML = `<option value="">Any activity</option>` +
        activities.map(a => `<option value="${escapeHTML(a)}">${escapeHTML(a)}</option>`).join('');
    $('#model-activity').innerHTML = `<option value="">Any</option>` +
        modelActivities.map(a => `<option value="${escapeHTML(a)}">${escapeHTML(a)}</option>`).join('');
    for (const [key, sel] of Object.entries(CONTROLS)) $(sel).value = state[key];

    $('#source-note').textContent = state.source === 'current'
        ? 'The main-task clips, including ones nobody has annotated yet, with the answers saved in this app. ' +
          'The model\'s label is shown where the clip was in the model\'s set.'
        : 'Every clip the model labeled. Clips checked in an earlier pass show those answers: the model validation ' +
          'pass (DD/JY/VM, description accuracy 1–5) and the original doing/seeing pass (confidence 1–3). Older labels ' +
          'count toward every current activity they could mean, e.g. "playing" toward each "playing with…".';
}

// ---------------------------------------------------------------------------
// Answers
// ---------------------------------------------------------------------------

// Current activities a label (possibly from an older label set) could stand for
const mapped = (label) => data.labelMap[label] || [];

function summarize(votes) {
    const counts = new Map();
    for (const v of votes) counts.set(v.primaryActivity, (counts.get(v.primaryActivity) || 0) + 1);
    const sorted = [...counts].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
    const withInteraction = votes.filter(v => ['yes', 'no'].includes(v.anyoneInteracting));
    const yes = withInteraction.filter(v => v.anyoneInteracting === 'yes').length;
    return {
        n: votes.length,
        counts: sorted,
        agreement: votes.length ? sorted[0][1] / votes.length : null,
        nInteraction: withInteraction.length,
        yes,
        forFocus: state.focus ? votes.filter(v => mapped(v.primaryActivity).includes(state.focus)).length : 0,
        mentionsFocus: state.focus ? votes.filter(v => [v.primaryActivity, ...(v.otherActivities || [])]
            .some(l => mapped(l).includes(state.focus))).length : 0
    };
}

function matchesCurrent(clip) {
    const s = clip.summary;
    if (s.n < +state.minVotes) return false;
    if (state.hideGold && clip.gold.some(g => g.modality === state.task)) return false;
    if (state.search) {
        const q = state.search.toLowerCase();
        if (!clip.videoFilename.toLowerCase().includes(q) &&
            !clip.votes.some(v => v.annotatorName.toLowerCase().includes(q))) return false;
    }
    if (state.agreement === 'split' && !(s.n >= 2 && s.agreement < 1)) return false;
    if (state.agreement === 'low' && !(s.n >= 2 && s.agreement <= 0.5)) return false;
    if (state.interaction === 'split' && !(s.yes > 0 && s.yes < s.nInteraction)) return false;
    if (state.interaction === 'yes' && !(s.yes > s.nInteraction / 2)) return false;
    if (state.interaction === 'no' && !(s.nInteraction > 0 && s.yes < s.nInteraction / 2)) return false;
    if (!state.focus) return true;
    switch (state.relCurrent) {
        case 'near': return s.mentionsFocus > 0 && s.forFocus <= s.n / 2;
        case 'mention': return s.mentionsFocus > 0;
        case 'majority': return s.forFocus > s.n / 2;
        case 'model': return !!clip.model && mapped(clip.model.activity).includes(state.focus) && s.forFocus <= s.n / 2;
        default: return true;
    }
}

function sortCurrent(list) {
    const byName = (a, b) => a.videoFilename.localeCompare(b.videoFilename);
    const sorters = {
        agreement: (a, b) => (a.summary.agreement ?? Infinity) - (b.summary.agreement ?? Infinity) ||
            b.summary.n - a.summary.n || byName(a, b),
        focus: (a, b) => b.summary.forFocus - a.summary.forFocus || b.summary.mentionsFocus - a.summary.mentionsFocus || byName(a, b),
        votes: (a, b) => b.summary.n - a.summary.n || byName(a, b),
        name: byName
    };
    return list.sort(sorters[state.sortCurrent] || sorters.agreement);
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

async function load({ append = false } = {}) {
    const id = ++requestId;
    writeURL();
    if (!append) {
        $$('#grid video').forEach(video => { lazyVideos.unobserve(video); releaseVideo(video); });
        $('#grid').innerHTML = '';
        $('#count').textContent = 'Loading…';
        $('#more').hidden = true;
    }

    if (state.source === 'current') {
        if (!data || data.source !== 'current' || data.taskType !== state.task) {
            const result = await api(`/admin/gallery?source=current&taskType=${state.task}`);
            if (id !== requestId) return;
            data = result;
        }
        syncControls();
        data.clips.forEach(c => { c.summary = summarize(c.votes); });
        clips = sortCurrent(data.clips.filter(matchesCurrent));
        shown = 0;
        const annotated = data.clips.filter(c => c.votes.length).length;
        $('#count').textContent = `${clips.length} of ${data.clips.length} clips · ${annotated} annotated so far`;
        showMore();
        return;
    }

    const params = new URLSearchParams({
        source: 'model', taskType: state.task, focus: state.focus, relation: state.relModel,
        checked: state.checked, modelActivity: state.modelActivity, sort: state.sortModel,
        search: state.search, hideGold: state.hideGold, seed: state.seed,
        offset: append ? clips.length : 0, limit: PAGE_SIZE
    });
    const result = await api(`/admin/gallery?${params}`);
    if (id !== requestId) return;
    modelActivities = result.modelActivities;
    if (append) {
        Object.assign(data.labelMap, result.labelMap);
        data.total = result.total;
    } else {
        data = result;
        clips = [];
        shown = 0;
        syncControls();
    }
    clips.push(...result.clips.map(c => ({ ...c, summary: summarize(c.earlier) })));
    $('#count').textContent = `${data.total.toLocaleString()} clips match · showing ${clips.length.toLocaleString()}`;
    showMore();
    $('#more').hidden = clips.length >= data.total;
}

function showMore() {
    const next = clips.slice(shown, shown + PAGE_SIZE);
    $('#grid').insertAdjacentHTML('beforeend', next.map(cardHTML).join(''));
    next.forEach((clip, i) => bindCard($(`.clip-card[data-i="${shown + i}"]`), clip));
    shown += next.length;
    if (!shown) $('#grid').innerHTML = '<div class="card center"><p>No clips match these filters.</p></div>';
    if (state.source === 'current') $('#more').hidden = shown >= clips.length;
}

// ---------------------------------------------------------------------------
// Cards
// ---------------------------------------------------------------------------

function voteChips(summary) {
    const top = summary.counts[0]?.[1];
    return summary.counts.map(([label, count]) => {
        const cls = [count === top && summary.n > 1 ? 'top' : '', state.focus && mapped(label).includes(state.focus) ? 'focus' : '']
            .join(' ');
        return `<span class="vote ${cls}">${escapeHTML(label)}<b>×${count}</b></span>`;
    }).join('');
}

function statsHTML(clip) {
    const s = clip.summary;
    const parts = [];
    if (state.source === 'current') {
        parts.push(s.n ? `<span class="stat">${s.n} answer${s.n === 1 ? '' : 's'}</span>` : '<span class="stat">Not annotated yet</span>');
        if (s.n >= 2) parts.push(`<span class="stat ${level(s.agreement)}">${pct(s.agreement)} agree</span>`);
    } else if (clip.validationScore == null) {
        parts.push('<span class="stat">Never checked</span>');
    } else {
        parts.push(`<span class="stat ${level(clip.validationScore)}" title="Mean of description accuracy (scaled) and the share of earlier answers that include the model's activity">Validated ${pct(clip.validationScore)}</span>`);
    }
    if (s.nInteraction) {
        const split = s.yes > 0 && s.yes < s.nInteraction;
        parts.push(`<span class="stat ${split ? 'mid' : ''}">Interacting: yes ${s.yes}/${s.nInteraction}</span>`);
    }
    clip.gold.forEach((g, i) => {
        // gold/example rows can be switched off (and back on) without losing their type
        const toggle = ['gold', 'example'].includes(g.type.trim().toLowerCase())
            ? `<button type="button" class="chip-toggle" data-gold-toggle="${i}">${g.disabled ? 'use again' : 'stop using'}</button>` : '';
        parts.push(`<span class="stat gold ${g.disabled ? 'off' : ''}" title="gold_standards.csv row ${g.csvRow}${g.disabled ? ' (turned off: not used by the app)' : ''}">★ ${escapeHTML(g.type)}${g.disabled ? ' (off)' : ''} · ${escapeHTML(g.modality)} · ${escapeHTML(g.primaryActivity)}${g.negativeOf ? ` (not ${escapeHTML(g.negativeOf)})` : ''}${toggle}</span>`);
    });
    return parts.join('');
}

function modelHTML(clip) {
    if (!clip.model) return '';
    const answers = state.source === 'model' ? clip.earlier : clip.votes;
    const confirmed = answers.filter(a => [a.primaryActivity, ...(a.otherActivities || [])]
        .some(l => l === clip.model.activity || mapped(l).some(m => mapped(clip.model.activity).includes(m)))).length;
    const check = !answers.length ? '' : confirmed
        ? ` · <span class="confirmed">${confirmed}/${answers.length} coders agree</span>`
        : ` · <span class="unconfirmed">no coder agrees</span>`;
    const focus = state.focus && mapped(clip.model.activity).includes(state.focus) ? ' focus' : '';
    return `
        <div class="model-line">Model: <span class="vote${focus}">${escapeHTML(clip.model.activity || '—')}</span>
            ${clip.model.location ? `· ${escapeHTML(clip.model.location)}` : ''}${check}</div>
        ${clip.model.description ? `<p class="description" title="Click to show all">${escapeHTML(clip.model.description)}</p>` : ''}`;
}

function answersTable(answers) {
    if (!answers.length) return '';
    const earlier = state.source === 'model';
    const rows = answers.map(a => `
        <tr>
            ${earlier ? `<td>${a.pass === 'validation' ? 'Validation' : 'Original'}</td>` : ''}
            <td>${escapeHTML(a.annotatorName)}</td>
            <td>${escapeHTML(a.primaryActivity)}${a.primaryActivityOther ? `: ${escapeHTML(a.primaryActivityOther)}` : ''}</td>
            ${earlier ? `<td>${escapeHTML((a.otherActivities || []).join(', '))}</td>
                <td>${a.accuracy != null ? `acc ${a.accuracy}/5` : a.confidence != null ? `conf ${a.confidence}/3` : ''}</td>` : ''}
            <td>${escapeHTML(a.anyoneInteracting || '')}</td>
        </tr>`).join('');
    return `
        <details><summary>${earlier ? 'Earlier answers' : 'Answers'} (${answers.length})</summary>
            <table class="answers">
                <tr>${earlier ? '<th>Pass</th>' : ''}<th>Coder</th><th>Activity</th>${earlier ? '<th>Also</th><th>Rating</th>' : ''}<th>Interacting</th></tr>
                ${rows}
            </table>
        </details>`;
}

function cardHTML(clip, i) {
    const index = shown + i;
    const answers = state.source === 'model' ? clip.earlier : clip.votes;
    const name = clip.videoFilename || clip.key;
    return `
        <div class="clip-card" data-i="${index}">
            <div class="video-frame">
                ${clip.src ? `<video data-src="${escapeHTML(clip.src)}" controls preload="none" playsinline muted></video>` : ''}
                <div class="video-placeholder" ${clip.src ? 'hidden' : ''}><strong>Video not available</strong><span>${escapeHTML(name)}</span></div>
            </div>
            <div class="clip-body">
                <div class="clip-name">${escapeHTML(name)}${clip.oldVideoFilename && clip.oldVideoFilename !== name ? `<br>was ${escapeHTML(clip.oldVideoFilename)}` : ''}</div>
                <div class="clip-stats">${statsHTML(clip)}</div>
                ${clip.summary.n ? `<div class="votes">${voteChips(clip.summary)}</div>` : ''}
                ${modelHTML(clip)}
                ${answersTable(answers)}
                <div class="add-slot"><button type="button" class="btn btn-small" data-add ${clip.src ? '' : 'disabled'}>＋ Add to gold standards</button></div>
            </div>
        </div>`;
}

// Videos load when the card scrolls into view and are released when it scrolls away (unless playing).
// Videos kept loaded tie up the browser's few connections to the server, and later ones never load.
const lazyVideos = new IntersectionObserver(entries => {
    for (const { target: video, isIntersecting } of entries) {
        if (isIntersecting && !video.getAttribute('src')) {
            video.src = video.dataset.src;
            video.preload = 'metadata';
        } else if (!isIntersecting && video.paused && video.getAttribute('src')) {
            releaseVideo(video);
        }
    }
}, { rootMargin: '400px' });

function releaseVideo(video) {
    video.pause();
    video.removeAttribute('src');
    video.load();
}

function bindCard(el, clip) {
    const video = $('video', el);
    if (video) {
        video.addEventListener('error', () => {
            if (!video.getAttribute('src')) return;
            video.hidden = true;
            video.nextElementSibling.hidden = false;
        });
        lazyVideos.observe(video);
    }
    $('[data-add]', el).addEventListener('click', () => openAddForm(el, clip));
}

// ---------------------------------------------------------------------------
// Add to gold standards
// ---------------------------------------------------------------------------

function defaultsFor(clip) {
    const answers = state.source === 'model' ? clip.earlier : clip.votes;
    const list = data.activities[state.task];
    const top = clip.summary.counts[0]?.[0];
    const topMapped = top ? mapped(top) : [];
    const s = clip.summary;
    const negativeRelation = state.source === 'model' ? state.relModel === 'model-only'
        : ['near', 'model'].includes(state.relCurrent);
    return {
        primary_activity: topMapped.length === 1 && list.includes(topMapped[0]) ? topMapped[0] : '',
        anyone_interacting: !s.nInteraction ? '' : s.yes > s.nInteraction / 2 ? 'yes' : 'no',
        negative_of: state.focus && negativeRelation && !topMapped.includes(state.focus) ? state.focus : '',
        hasAnswers: answers.length > 0
    };
}

const activityOptions = (list, selected, blank) =>
    `<option value="">${blank}</option>` +
    list.map(a => `<option value="${escapeHTML(a)}" ${a === selected ? 'selected' : ''}>${escapeHTML(a)}</option>`).join('');

function openAddForm(el, clip) {
    const d = defaultsFor(clip);
    const slot = $('.add-slot', el);
    slot.innerHTML = `
        <form class="add-form">
            <label>Task
                <select name="modality">
                    ${['do', 'see'].map(t => `<option value="${t}" ${t === state.task ? 'selected' : ''}>${taskName(t)}</option>`).join('')}
                </select>
            </label>
            <label>Type
                <select name="type">
                    <option value="candidate" selected>candidate (shortlist; not used yet)</option>
                    ${state.source === 'current' ? '' : `<option value="gold">gold (practice pool)</option>
                    <option value="example">example (shown with answers)</option>`}
                    <option value="confusing">confusing</option>
                </select>
            </label>
            <label>Correct activity
                <select name="primary_activity">${activityOptions(data.activities[state.task], d.primary_activity, 'Choose…')}</select>
            </label>
            <label>Anyone interacting
                <select name="anyone_interacting">
                    <option value="">Choose…</option>
                    ${['yes', 'no'].map(v => `<option ${v === d.anyone_interacting ? 'selected' : ''}>${v}</option>`).join('')}
                </select>
            </label>
            <label class="full">Negative example of
                <select name="negative_of">${activityOptions(data.activities[state.task], d.negative_of, '— (not a negative example)')}</select>
            </label>
            <label class="full">Primary activity reasoning
                <textarea name="pa_reasoning" placeholder="Why this is the correct activity${d.negative_of ? ` and not ${escapeHTML(d.negative_of)}` : ''}…"></textarea>
            </label>
            <label class="full">Interaction reasoning
                <textarea name="ai_reasoning"></textarea>
            </label>
            <div class="full hint">
                ${state.source === 'current'
                    ? 'This is a main-task clip, so it can only be a candidate or confusing: as gold or example, coders would see the answer before annotating it.'
                    : 'Gold and example rows change what coders see in training right away, and adding a gold clip reshuffles practice sets for coders still in training. Add as <em>candidate</em> to shortlist it, then change its type in the gold standards editor.'}
            </div>
            <div class="full" style="display: flex; gap: 8px;">
                <button type="submit" class="btn btn-primary btn-small">Add</button>
                <button type="button" class="btn btn-small" data-cancel>Cancel</button>
            </div>
        </form>`;
    const form = $('form', slot);

    form.modality.addEventListener('change', () => {
        const list = data.activities[form.modality.value];
        form.primary_activity.innerHTML = activityOptions(list, form.primary_activity.value, 'Choose…');
        form.negative_of.innerHTML = activityOptions(list, form.negative_of.value, '— (not a negative example)');
    });
    $('[data-cancel]', form).addEventListener('click', () => {
        slot.innerHTML = '<button type="button" class="btn btn-small" data-add>＋ Add to gold standards</button>';
        $('[data-add]', slot).addEventListener('click', () => openAddForm(el, clip));
    });
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!form.primary_activity.value || !form.anyone_interacting.value) {
            toast('Choose the correct activity and the interaction answer', true);
            return;
        }
        if (form.negative_of.value && form.negative_of.value === form.primary_activity.value) {
            toast('A clip can\'t be a negative example of its own activity', true);
            return;
        }
        const button = $('[type=submit]', form);
        button.disabled = true;
        try {
            const result = await api('/admin/gold', {
                method: 'POST',
                body: JSON.stringify({
                    source: state.source,
                    key: clip.key,
                    video_filename: clip.videoFilename,
                    modality: form.modality.value,
                    type: form.type.value,
                    primary_activity: form.primary_activity.value,
                    anyone_interacting: form.anyone_interacting.value,
                    negative_of: form.negative_of.value,
                    pa_reasoning: form.pa_reasoning.value,
                    ai_reasoning: form.ai_reasoning.value
                })
            });
            clip.gold.push({
                csvRow: result.csvRow, videoFilename: result.videoFilename, type: form.type.value,
                modality: form.modality.value, primaryActivity: form.primary_activity.value,
                negativeOf: form.negative_of.value, disabled: false
            });
            $('.clip-stats', el).innerHTML = statsHTML(clip);
            slot.innerHTML = `<span class="saved-note" style="color: var(--good); font-size: 13px;">✓ Added as row ${result.csvRow} of gold_standards.csv</span>`;
            toast(`Added ${result.videoFilename} as row ${result.csvRow}`);
        } catch (error) {
            toast('Could not add: ' + error.message, true);
            button.disabled = false;
        }
    });
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

function reload() {
    load().catch(err => {
        $('#count').textContent = '';
        $('#grid').innerHTML = `<div class="card center"><p class="error">${escapeHTML(err.message)}</p></div>`;
    });
}

$$('#source button').forEach(b => b.addEventListener('click', () => { state.source = b.dataset.value; syncControls(); reload(); }));
$$('#task button').forEach(b => b.addEventListener('click', () => { state.task = b.dataset.value; reload(); }));
for (const [key, sel] of Object.entries(CONTROLS)) {
    if (key === 'search') continue;
    $(sel).addEventListener('change', () => { state[key] = $(sel).value; reload(); });
}
let searchTimer;
$('#search').addEventListener('input', () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => { state.search = $('#search').value.trim(); reload(); }, 300);
});
$('#grid').addEventListener('click', async (e) => {
    const description = e.target.closest('.description');
    if (description) description.classList.toggle('open');

    const btn = e.target.closest('[data-gold-toggle]');
    if (!btn) return;
    const el = btn.closest('.clip-card');
    const clip = clips[+el.dataset.i];
    const g = clip.gold[+btn.dataset.goldToggle];
    btn.disabled = true;
    try {
        await api(`/admin/gold/${g.csvRow - 2}`, {
            method: 'POST',
            body: JSON.stringify({ video_filename: g.videoFilename, disabled: g.disabled ? '' : 'yes' })
        });
        g.disabled = !g.disabled;
        $('.clip-stats', el).innerHTML = statsHTML(clip);
        toast(`Row ${g.csvRow} ${g.disabled ? 'is no longer used by the app' : `is used again as ${g.type} for ${taskName(g.modality).toLowerCase()}`}`);
    } catch (error) {
        toast(error.message, true);
        btn.disabled = false;
    }
});
$('#hide-gold').addEventListener('change', () => { state.hideGold = $('#hide-gold').checked; reload(); });
$('#reshuffle').addEventListener('click', () => { state.seed = Math.random().toString(36).slice(2, 8); reload(); });
$('#more').addEventListener('click', () => {
    if (state.source === 'current') showMore();
    else load({ append: true }).catch(err => toast(err.message, true));
});

readURL();
syncControls();
reload();
