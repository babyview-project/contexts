// activities.js - activity annotation flow
// Phases: examples -> practice (graded against gold standards) -> main (assigned clips) -> done
// URL: activities.html?type=do | ?type=see (without it the coder picks a task)

const TASK_LABELS = { do: 'Doing', see: 'Seeing' };
const ID_STORAGE_KEY = 'activity_annotations_last_id';

// ?dev=1 asks for dev mode (skip controls, relaxed training gates); the server only grants it to
// admin accounts (or to everyone when there are no accounts)
const DEV_REQUESTED = ['1', 'true'].includes(new URLSearchParams(location.search).get('dev'));

// Current URL with the task type changed, keeping other parameters such as dev=1
function urlWithType(type) {
    const params = new URLSearchParams(location.search);
    params.set('type', type);
    return `?${params}`;
}

const state = {
    config: null,
    taskType: null,
    annotator: '',
    resolveScreen: null,
    pendingJump: null,
    // Dev mode records nothing on the server, so its practice attempts and main-task answers live here
    devAttempts: 0,
    devAnnotations: {}
};

const DEV_SANDBOX_NOTE = 'Dev mode: nothing here is saved, and clips are not assigned to you.';

// Seed for the dev-mode preview clip set, remembered per tab and task so a reload keeps the same clips
const PREVIEW_SEED_KEY = 'activity_annotations_preview_seed';
function getPreviewSeed() {
    try { return sessionStorage.getItem(`${PREVIEW_SEED_KEY}_${state.taskType}`) || ''; } catch (e) { return ''; }
}
function newPreviewSeed() {
    const seed = Math.random().toString(36).slice(2, 10);
    try { sessionStorage.setItem(`${PREVIEW_SEED_KEY}_${state.taskType}`, seed); } catch (e) { /* storage unavailable */ }
    return seed;
}
const assignmentURL = () =>
    `/assignment?taskType=${state.taskType}&previewSeed=${encodeURIComponent(getPreviewSeed())}`;

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

function escapeHTML(value) {
    return String(value ?? '').replace(/[&<>"']/g, c =>
        ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

const pct = (x) => `${Math.round(x * 100)}%`;
const activities = () => state.config.activities[state.taskType];
const definitionOf = (label) => state.config.activities.definitions[state.taskType]?.[label] || '';

function toast(message, isError = false) {
    const el = $('#toast');
    el.textContent = message;
    el.classList.toggle('error', isError);
    el.classList.add('show');
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => el.classList.remove('show'), isError ? 5000 : 2500);
}

// ============================================================================
// API
// ============================================================================

async function api(path, { method = 'GET', body, raw = false } = {}) {
    const headers = { 'Content-Type': 'application/json' };
    if (state.annotator) headers['X-Annotator-Id'] = encodeURIComponent(state.annotator);
    if (DEV_REQUESTED) headers['X-Dev-Mode'] = '1';

    const response = await fetch('/api' + path, {
        method,
        credentials: 'include',
        headers,
        body: body === undefined ? undefined : JSON.stringify(body)
    });

    // Session expired: log in again, then retry the same request.
    // Logging in as someone else mid-task restarts the page so their own progress loads.
    if (response.status === 401 && state.config?.authMode === 'accounts') {
        const user = await showLogin('Your session has expired. Please log in again.');
        if (state.annotator && user.username !== state.annotator) {
            location.reload();
            return new Promise(() => {});
        }
        return api(path, { method, body, raw });
    }
    if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        const error = new Error(err.error || `Request failed (${response.status})`);
        error.status = response.status;
        throw error;
    }
    return raw ? response : response.json();
}

// Resolves with the logged-in user ({ username, admin })
function showLogin(message = '') {
    const modal = $('#login-modal');
    const form = $('#login-form');
    $('#login-message').textContent = message;
    $('#login-error').hidden = true;
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
            const result = await res.json().catch(() => ({}));
            if (!res.ok) {
                $('#login-error').textContent = result.error || 'Login failed. Please try again.';
                $('#login-error').hidden = false;
                return;
            }
            modal.hidden = true;
            resolve(result.user);
        };
    });
}

// ============================================================================
// SCREENS
// ============================================================================

// Renders a screen and resolves with whatever the screen's buttons pass to `done`.
// Dev-mode skip buttons resolve the current screen with { jump: phase }.
function showScreen(html, setup) {
    if (state.pendingJump) {
        const jump = state.pendingJump;
        state.pendingJump = null;
        return Promise.resolve({ jump });
    }
    return new Promise(resolve => {
        const app = $('#app');
        // A removed <video> keeps downloading until it is garbage collected, holding one of the
        // browser's few connections to the server; clicking through clips quickly then leaves the
        // next clip waiting to load. Stop the old ones first.
        $$('video', app).forEach(video => {
            video.pause();
            video.removeAttribute('src');
            video.load();
        });
        app.innerHTML = html;
        window.scrollTo(0, 0);
        const done = (value) => {
            if (state.resolveScreen !== done) return;
            state.resolveScreen = null;
            resolve(value);
        };
        state.resolveScreen = done;
        bindVideoFallbacks(app);
        $$('[data-done]', app).forEach(btn =>
            btn.addEventListener('click', () => done(btn.dataset.done)));
        $$('[data-open-definitions]', app).forEach(btn =>
            btn.addEventListener('click', () => openDefinitions(btn.dataset.openDefinitions)));
        if (setup) setup(app, done);
    });
}

function messageScreen(title, body, buttons = [['Continue', 'continue', true]]) {
    return showScreen(`
        <div class="card center">
            <h2>${title}</h2>
            ${body}
            <div class="actions">
                ${buttons.map(([label, value, primary]) =>
                    `<button type="button" class="btn ${primary ? 'btn-primary' : ''}" data-done="${value}">${label}</button>`
                ).join('')}
            </div>
        </div>`);
}

function setStep(text) {
    $('#step').textContent = text || '';
}

function videoHTML(src, filename, { autoplay = true } = {}) {
    return `
        <div class="video-frame">
            <video src="${escapeHTML(src)}" controls playsinline ${autoplay ? 'autoplay' : 'preload="metadata"'}></video>
            <div class="video-placeholder" hidden>
                <strong>${state.config.testMode ? 'Test mode: video placeholder' : 'Video could not be loaded'}</strong>
                <span>${escapeHTML(filename)}</span>
            </div>
        </div>`;
}

// Missing video files (always the case in test mode) show a placeholder instead of a black box
function bindVideoFallbacks(root) {
    $$('.video-frame video', root).forEach(video => {
        video.addEventListener('error', () => {
            video.hidden = true;
            video.nextElementSibling.hidden = false;
        });
    });
}

const trainingVideoURL = (f) => `/training-videos/${encodeURIComponent(f)}`;
const mainVideoURL = (f) => `/videos/${encodeURIComponent(f)}`;

// ============================================================================
// DEFINITIONS MODAL
// ============================================================================

// section: 'activity' | 'interacting' | 'all'
function renderDefinitions(taskType, section = 'all') {
    const { definitions, anyoneInteracting, taskDescriptions } = state.config.activities;
    const verb = taskType === 'see' ? 'seeing' : 'doing';
    $$('#definitions-modal [data-tab]').forEach(tab =>
        tab.setAttribute('aria-selected', tab.dataset.tab === taskType));
    $('#definitions-title').textContent = {
        activity: `Activities the child is ${verb}`,
        interacting: 'Is anyone interacting with the child?',
        all: 'Definitions'
    }[section];

    const activityHTML = `
        <p class="muted">${escapeHTML(taskDescriptions[taskType])}</p>
        <dl class="definitions">
            ${state.config.activities[taskType].map(label => `
                <dt>${escapeHTML(label)}</dt>
                <dd>${escapeHTML(definitions[taskType]?.[label] || '')}</dd>`).join('')}
        </dl>`;
    const interactingHTML = `<p>${escapeHTML(anyoneInteracting)}</p>`;

    $('#definitions-body').innerHTML =
        section === 'activity' ? activityHTML :
        section === 'interacting' ? interactingHTML : `
            <h3>1. Primary activity the child is ${verb}</h3>${activityHTML}
            <h3 style="margin-top: 24px;">2. Is anyone interacting with the child?</h3>${interactingHTML}`;
}

// Once a task is chosen only its definitions are shown; tabs are only offered before that
function openDefinitions(section = 'all') {
    $('#definitions-modal .tabs').hidden = !!state.taskType;
    renderDefinitions(state.taskType || 'do', section);
    $('#definitions-modal').hidden = false;
    $('#definitions-modal .icon-btn').focus();
}

function setupChrome() {
    // The top bar wraps on narrower windows; the video/panel height is sized from its real height
    const topbar = $('.topbar');
    const syncTopbarHeight = () =>
        document.documentElement.style.setProperty('--topbar-h', `${topbar.offsetHeight}px`);
    new ResizeObserver(syncTopbarHeight).observe(topbar);
    syncTopbarHeight();

    const modal = $('#definitions-modal');
    $('#info-btn').addEventListener('click', () => openDefinitions('all'));
    $$('[data-close]', modal).forEach(el => el.addEventListener('click', () => { modal.hidden = true; }));
    $$('[data-tab]', modal).forEach(tab => tab.addEventListener('click', () => renderDefinitions(tab.dataset.tab)));
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') modal.hidden = true;
    });

    showModeBadge();
    $('#logout-btn').addEventListener('click', async () => {
        if (!confirm('Log out? Your saved annotations are kept.')) return;
        await fetch('/api/auth/logout', { method: 'POST', credentials: 'include' });
        location.reload();
    });
    $$('#dev-nav [data-jump]').forEach(btn => btn.addEventListener('click', () => {
        if (state.resolveScreen) state.resolveScreen({ jump: btn.dataset.jump });
        else state.pendingJump = btn.dataset.jump;
    }));
}

function showModeBadge() {
    const { devMode, testMode } = state.config;
    const badge = $('#mode-badge');
    badge.textContent = [testMode && 'TEST MODE', devMode && 'DEV MODE'].filter(Boolean).join(' · ');
    badge.hidden = !(devMode || testMode);
}

function showTaskBadges() {
    const badge = $('#task-badge');
    const other = state.taskType === 'do' ? 'see' : 'do';
    badge.innerHTML = `Task: ${TASK_LABELS[state.taskType]}` +
        (state.config.devMode || state.config.testMode
            ? `<a href="${urlWithType(other)}" title="Restart with the other task">switch to ${TASK_LABELS[other]}</a>`
            : '');
    badge.hidden = false;
    if (state.annotator) {
        $('#annotator-badge').textContent = `ID: ${state.annotator}`;
        $('#annotator-badge').hidden = false;
    }
}

// ============================================================================
// ACTIVITY COMBOBOX - type to filter, arrows + Enter/Tab or click to pick
// ============================================================================

// Only values from `options` can be chosen; unmatched text is reverted on blur.
// Returns an object whose `value` is the chosen option ('' if none).
function createCombobox(wrapper, options, initial, onChange) {
    const input = $('input', wrapper);
    const pop = $('.combobox-pop', wrapper);
    const list = $('ul', wrapper);
    const preview = $('.combobox-preview', wrapper);
    let selected = initial;
    let matches = [];
    let active = -1;

    // Options starting with the query (or with a word starting with it) come first
    const filter = (query) => {
        const q = query.trim().toLowerCase();
        if (!q) return [...options];
        const rank = (o) => o.startsWith(q) ? 0 : o.split(/[\s/]+/).some(w => w.startsWith(q)) ? 1 : 2;
        return options.filter(o => o.includes(q)).sort((a, b) => rank(a) - rank(b));
    };

    const highlight = (option, query) => {
        const q = query.trim().toLowerCase();
        const i = q ? option.indexOf(q) : -1;
        if (i < 0) return escapeHTML(option);
        return escapeHTML(option.slice(0, i)) + '<mark>' + escapeHTML(option.slice(i, i + q.length)) +
            '</mark>' + escapeHTML(option.slice(i + q.length));
    };

    const render = (query) => {
        matches = filter(query);
        active = matches.length ? Math.max(0, matches.indexOf(selected)) : -1;
        list.innerHTML = matches.length
            ? matches.map((o, i) => `
                <li role="option" id="combo-opt-${i}" data-index="${i}" aria-selected="${o === selected}"
                    class="${i === active ? 'active' : ''}">${highlight(o, query)}</li>`).join('')
            : '<li class="combobox-empty">No matching activity</li>';
        pop.hidden = false;
        setActive(active);
        // Show every option: switch to two columns if one column won't fit in the window,
        // and only as a last resort (very short windows) let the list scroll
        const room = window.innerHeight - pop.getBoundingClientRect().top - 8;
        list.classList.remove('two-col');
        list.style.maxHeight = '';
        if (pop.offsetHeight > room) list.classList.add('two-col');
        if (pop.offsetHeight > room) {
            list.style.maxHeight = `${Math.max(room - preview.offsetHeight, 120)}px`;
        }
        input.setAttribute('aria-expanded', 'true');
    };

    // The highlighted option (keyboard or mouse hover) gets its definition shown below the list
    const setActive = (i, { scroll = true } = {}) => {
        active = i;
        $$('[role=option]', list).forEach(li => li.classList.toggle('active', +li.dataset.index === i));
        const li = $(`[data-index="${i}"]`, list);
        const option = matches[i];
        preview.innerHTML = option
            ? `<strong>${escapeHTML(option)}:</strong> ${escapeHTML(definitionOf(option))}`
            : '';
        preview.hidden = !option;
        if (li) {
            if (scroll) li.scrollIntoView({ block: 'nearest' });
            input.setAttribute('aria-activedescendant', li.id);
        } else {
            input.removeAttribute('aria-activedescendant');
        }
    };

    const close = () => {
        pop.hidden = true;
        input.setAttribute('aria-expanded', 'false');
        input.removeAttribute('aria-activedescendant');
    };

    const choose = (option) => {
        selected = option;
        input.value = option;
        input.select();
        close();
        onChange(option);
    };

    // On blur: accept an exact (case-insensitive) match, otherwise restore the last choice
    const commitTyped = () => {
        const typed = input.value.trim().toLowerCase();
        const exact = options.find(o => o === typed);
        if (exact) {
            if (exact !== selected) choose(exact);
            else input.value = exact;
        } else {
            input.value = selected;
        }
        close();
    };

    // Select the current value on focus so typing replaces it; the mouseup that follows a
    // click-to-focus would otherwise drop the selection and place the caret inside the text
    let justFocused = false;
    input.addEventListener('focus', () => { justFocused = true; input.select(); render(''); });
    input.addEventListener('mouseup', (e) => {
        if (justFocused) e.preventDefault();
        justFocused = false;
    });
    input.addEventListener('keydown', () => { justFocused = false; }, true);
    // Clicking an already-focused, closed input (e.g. right after choosing) reopens the list with
    // the text selected; preventing mousedown stops the browser from collapsing the selection
    input.addEventListener('mousedown', (e) => {
        if (document.activeElement !== input || !pop.hidden) return;
        e.preventDefault();
        input.select();
        render('');
    });
    input.addEventListener('input', () => render(input.value));
    input.addEventListener('blur', commitTyped);
    input.addEventListener('keydown', (e) => {
        const open = !pop.hidden;
        if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
            e.preventDefault();
            if (!open) return render('');
            if (!matches.length) return;
            const step = e.key === 'ArrowDown' ? 1 : -1;
            setActive((active + step + matches.length) % matches.length);
        } else if (e.key === 'Enter') {
            if (open && active >= 0) {
                e.preventDefault(); // pick the option instead of submitting the form
                choose(matches[active]);
            }
        } else if (e.key === 'Tab') {
            if (open && active >= 0 && input.value.trim() && input.value !== selected) choose(matches[active]);
        } else if (e.key === 'Escape') {
            if (open) {
                e.stopPropagation();
                input.value = selected;
                close();
            }
        }
    });

    list.addEventListener('mouseover', (e) => {
        const li = e.target.closest('[role=option]');
        if (li && +li.dataset.index !== active) setActive(+li.dataset.index, { scroll: false });
    });
    // mousedown (not click) so the input doesn't blur first
    pop.addEventListener('mousedown', (e) => {
        e.preventDefault();
        const li = e.target.closest('[role=option]');
        if (li) choose(matches[+li.dataset.index]);
    });
    $('.combobox-toggle', wrapper).addEventListener('mousedown', (e) => {
        e.preventDefault();
        if (pop.hidden) { input.focus(); render(''); } else close();
    });

    return { get value() { return selected; } };
}

// ============================================================================
// ANNOTATION FORM (shared by practice and main task)
// ============================================================================

// Resolves with { action: 'submit', values } | { action: 'back' } | { action: 'goto', index } |
// { action: 'quit' } | { jump }
function annotationScreen({ videoURL, filename, existing, isPractice, index, total, completedCount, canGoBack }) {
    const isSeeing = state.taskType === 'see';
    const values = existing || {};

    const progressHTML = isPractice ? '' : `
        <div class="panel-section">
            <div class="progress-row">
                <strong>Clip ${index + 1} of ${total}</strong>
                <span class="status-pill ${existing ? 'done' : 'todo'}">${existing ? '✓ Annotated' : 'Not annotated yet'}</span>
            </div>
            <div class="progress-row muted">
                <span>Completed ${completedCount} of ${total}</span><span>${pct(completedCount / total)}</span>
            </div>
            <div class="progress-bar"><div class="progress-fill" style="width: ${(completedCount / total) * 100}%"></div></div>
            <div class="nav-row">
                <label for="goto-input" style="margin: 0; font-weight: 400;">Go to clip</label>
                <input type="number" id="goto-input" min="1" max="${total}" placeholder="${index + 1}">
                <button type="button" class="btn btn-small" id="goto-btn">Go</button>
                <span class="spacer"></span>
                <button type="button" class="btn btn-small btn-danger" id="quit-btn">Save &amp; quit</button>
            </div>
            ${state.config.devMode ? `
                <div class="nav-row">
                    <span class="muted" style="font-size: 12px; flex: 1;">${DEV_SANDBOX_NOTE}</span>
                    <button type="button" class="btn btn-small" id="shuffle-btn">🔀 Shuffle preview clips</button>
                </div>` : ''}
        </div>`;

    return showScreen(`
        <div class="workspace">
            <div class="video-col">${videoHTML(videoURL, filename)}</div>
            <div class="panel">
                ${isPractice ? `<div class="panel-section"><strong>Practice clip ${index + 1} of ${total}</strong></div>` : ''}
                ${progressHTML}
                <form class="panel-section" id="annotation-form" novalidate>
                    <div class="question">
                        <div class="question-title">
                            <label for="primary-select" style="margin: 0;">1. What is the primary activity that the child wearing the camera is ${isSeeing ? 'seeing' : 'doing'}?</label>
                            <button type="button" class="info-link" data-open-definitions="activity">ⓘ definitions</button>
                        </div>
                        <div class="combobox">
                            <input type="text" id="primary-select" role="combobox" autocomplete="off" spellcheck="false"
                                   aria-autocomplete="list" aria-expanded="false" aria-controls="primary-listbox"
                                   placeholder="Type or select primary activity…" value="${escapeHTML(values.primaryActivity || '')}">
                            <button type="button" class="combobox-toggle" tabindex="-1" aria-label="Show all activities">▾</button>
                            <div class="combobox-pop" hidden>
                                <ul id="primary-listbox" class="combobox-list" role="listbox"></ul>
                                <div class="combobox-preview" hidden></div>
                            </div>
                        </div>
                        <div class="definition-hint" id="primary-hint" hidden></div>
                        <div id="other-wrap" hidden style="margin-top: 10px;">
                            <label for="other-input" style="margin-bottom: 4px;">Please specify the primary activity${isSeeing ? ' seen' : ''}:</label>
                            <input type="text" id="other-input" value="${escapeHTML(values.primaryActivityOther || '')}">
                        </div>
                    </div>
                    <div class="question">
                        <div class="question-title">
                            <span>2. Is anyone interacting with the child?</span>
                            <button type="button" class="info-link" data-open-definitions="interacting">ⓘ definition</button>
                        </div>
                        <div class="radio-row">
                            ${['yes', 'no'].map(v => `
                                <label><input type="radio" name="interacting" value="${v}" ${values.anyoneInteracting === v ? 'checked' : ''}> ${v === 'yes' ? 'Yes' : 'No'}</label>
                            `).join('')}
                        </div>
                    </div>
                    <p class="error" id="form-error" hidden></p>
                    <div class="actions spread">
                        <button type="button" class="btn" id="back-btn" ${canGoBack ? '' : 'disabled'}>← Back</button>
                        <button type="submit" class="btn btn-primary" id="submit-btn">${isPractice && index + 1 === total ? 'Submit answers' : 'Save &amp; next →'}</button>
                    </div>
                </form>
            </div>
        </div>`, (root, done) => {
        const hint = $('#primary-hint', root);
        const otherWrap = $('#other-wrap', root);
        const otherInput = $('#other-input', root);
        const formError = $('#form-error', root);

        const syncPrimary = (value) => {
            const def = definitionOf(value);
            hint.textContent = def;
            hint.hidden = !def;
            otherWrap.hidden = value !== 'other';
        };
        const select = createCombobox($('.combobox', root), activities(), values.primaryActivity || '', syncPrimary);
        syncPrimary(select.value);

        $('#annotation-form', root).addEventListener('submit', (e) => {
            e.preventDefault();
            const interacting = $('input[name=interacting]:checked', root);
            const problems = [];
            if (!select.value) problems.push('choose a primary activity');
            if (select.value === 'other' && !otherInput.value.trim()) problems.push('describe the "other" activity');
            if (!interacting) problems.push('answer whether anyone is interacting with the child');
            if (problems.length) {
                formError.textContent = 'Please ' + problems.join(', ') + '.';
                formError.hidden = false;
                return;
            }
            done({
                action: 'submit',
                values: {
                    primaryActivity: select.value,
                    primaryActivityOther: select.value === 'other' ? otherInput.value.trim() : '',
                    anyoneInteracting: interacting.value
                }
            });
        });
        $('#back-btn', root).addEventListener('click', () => done({ action: 'back' }));

        if (!isPractice) {
            const gotoInput = $('#goto-input', root);
            const go = () => {
                const n = parseInt(gotoInput.value, 10);
                if (!(n >= 1 && n <= total)) {
                    toast(`Enter a clip number between 1 and ${total}`, true);
                    return;
                }
                done({ action: 'goto', index: n - 1 });
            };
            $('#goto-btn', root).addEventListener('click', go);
            gotoInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') { e.preventDefault(); go(); }
            });
            $('#shuffle-btn', root)?.addEventListener('click', () => done({ action: 'shuffle' }));
            $('#quit-btn', root).addEventListener('click', () => {
                if (confirm('Quit for now? Saved annotations are kept and you can resume later with the same ID.')) {
                    done({ action: 'quit' });
                }
            });
        }
    });
}

// ============================================================================
// PHASES - each returns the name of the next phase (or null to stop)
// ============================================================================

async function examplesPhase() {
    setStep('Training · Examples');
    const { videos } = await api(`/training/examples?taskType=${state.taskType}`);
    const { training } = state.config;

    let r = await showScreen(`
        <div class="card">
            <h2>Training</h2>
            <p>Before annotating, you'll go through a short training:</p>
            <ol>
                <li><strong>Examples:</strong> ${videos.length} clips shown with the correct annotations and the reasoning behind them.</li>
                <li><strong>Practice:</strong> you annotate ${training.videosPerAttempt} clips yourself. You need at least ${pct(training.passThreshold)} correct to pass (a clip is correct when both answers match).</li>
                <li><strong>Feedback:</strong> after each practice round you'll see the correct answers. If you don't pass, you get a new set of clips (up to ${training.maxAttempts} attempts).</li>
            </ol>
            <p class="muted">You can open the activity definitions at any time with the <strong>ⓘ Definitions</strong> button at the top of the page.</p>
            <div class="actions"><button type="button" class="btn btn-primary" data-done="start">Start examples</button></div>
        </div>`);
    if (r?.jump) return r.jump;

    if (videos.length === 0) {
        r = await messageScreen('No examples available', '<p>No example clips are configured for this task.</p>');
        return r?.jump || 'practice';
    }

    for (let i = 0; i < videos.length;) {
        const v = videos[i];
        setStep(`Training · Example ${i + 1} of ${videos.length}`);
        const isLast = i === videos.length - 1;
        const reasoningBlock = (id, text) => `
            <button type="button" class="btn btn-small" data-reveal="${id}">Reveal reasoning</button>
            <div class="reasoning" id="${id}" hidden>${escapeHTML(text || 'No reasoning provided for this clip.')}</div>`;

        r = await showScreen(`
            <div class="workspace">
                <div class="video-col">${videoHTML(trainingVideoURL(v.videoFilename), v.videoFilename)}</div>
                <div class="panel">
                    <div class="panel-section">
                        <strong>Example ${i + 1} of ${videos.length}</strong>
                        <p class="muted" style="margin: 4px 0 0;">Watch the clip, then reveal both reasonings to continue.</p>
                    </div>
                    <div class="panel-section">
                        <div class="answer-row">
                            <div class="answer-label">Primary activity the child is ${state.taskType === 'see' ? 'seeing' : 'doing'}</div>
                            <div class="answer-value">${escapeHTML(v.primaryActivity)}</div>
                            <p class="muted" style="font-size: 13px;">${escapeHTML(definitionOf(v.primaryActivity))}</p>
                            ${reasoningBlock('reason-primary', v.primaryActivityReasoning)}
                        </div>
                        <div class="answer-row">
                            <div class="answer-label">Is anyone interacting with the child?</div>
                            <div class="answer-value">${escapeHTML(v.anyoneInteracting)}</div>
                            ${reasoningBlock('reason-interacting', v.interactingReasoning)}
                        </div>
                    </div>
                    <p class="muted" id="reveal-hint" style="font-size: 13px; text-align: right; margin: 0;">Reveal both reasonings to continue</p>
                    <div class="actions spread" style="margin-top: 8px;">
                        <button type="button" class="btn" data-done="back" ${i === 0 ? 'disabled' : ''}>← Back</button>
                        <button type="button" class="btn btn-primary" id="next-btn" disabled>${isLast ? 'Continue to practice →' : 'Next example →'}</button>
                    </div>
                </div>
            </div>`, (root, done) => {
            const next = $('#next-btn', root);
            const update = () => {
                next.disabled = $$('.reasoning[hidden]', root).length > 0;
                $('#reveal-hint', root).hidden = !next.disabled;
            };
            $$('[data-reveal]', root).forEach(btn => btn.addEventListener('click', () => {
                $('#' + btn.dataset.reveal, root).hidden = false;
                btn.hidden = true;
                update();
            }));
            next.addEventListener('click', () => { if (!next.disabled) done('next'); });
            update();
        });

        if (r?.jump) return r.jump;
        i += r === 'back' ? -1 : 1;
    }
    return 'practice';
}

async function practicePhase() {
    const { devMode, training } = state.config;
    setStep('Training · Practice');
    const status = await api(`/training/status?taskType=${state.taskType}`);

    if (status.passed && !devMode) return 'main';
    if (status.attemptsRemaining === 0 && !devMode) return 'blocked';

    const attemptNumber = devMode ? state.devAttempts + 1 : status.attemptsUsed + 1;
    let r = await showScreen(`
        <div class="card center">
            <h2>Practice attempt ${attemptNumber} of ${training.maxAttempts}</h2>
            <p>Annotate the next ${training.videosPerAttempt} clips on your own. You need at least
               ${pct(training.passThreshold)} correct to pass. You'll see the correct answers afterwards.</p>
            ${devMode ? `<p class="muted">${DEV_SANDBOX_NOTE} Practice attempts are graded but not recorded.</p>` : ''}
            <div class="actions">
                <button type="button" class="btn" data-done="examples">Review examples</button>
                <button type="button" class="btn btn-primary" data-done="start">Start practice</button>
            </div>
        </div>`);
    if (r?.jump) return r.jump;
    if (r === 'examples') return 'examples';

    const { videos } = await api(`/training/practice-set?taskType=${state.taskType}&attempt=${attemptNumber}`);
    const responses = [];

    for (let i = 0; i < videos.length;) {
        setStep(`Practice ${attemptNumber}/${training.maxAttempts} · Clip ${i + 1} of ${videos.length}`);
        r = await annotationScreen({
            videoURL: trainingVideoURL(videos[i].videoFilename),
            filename: videos[i].videoFilename,
            existing: responses[i],
            isPractice: true,
            index: i,
            total: videos.length,
            canGoBack: i > 0
        });
        if (r?.jump) return r.jump;
        if (r.action === 'back') { i--; continue; }
        responses[i] = r.values;
        i++;
    }

    let result;
    try {
        result = await api('/training/attempts', {
            method: 'POST',
            body: {
                taskType: state.taskType,
                attemptNumber,
                responses: videos.map((v, i) => ({ videoFilename: v.videoFilename, ...responses[i] }))
            }
        });
    } catch (error) {
        toast('Could not submit practice answers: ' + error.message, true);
        return 'practice';
    }
    if (result.devMode) state.devAttempts = attemptNumber;
    return practiceFeedback(result);
}

async function practiceFeedback(result) {
    const { devMode, training } = state.config;
    const remaining = result.status.attemptsRemaining;
    setStep(`Practice ${result.attemptNumber}/${training.maxAttempts} · Results`);

    const cell = (ok, text) => `<td class="${ok ? 'right' : 'wrong'}">${ok ? '✓' : '✗'} ${escapeHTML(text)}</td>`;
    const items = result.results.map((res, idx) => {
        const yours = res.response.primaryActivity === 'other' && res.response.primaryActivityOther
            ? `other (${res.response.primaryActivityOther})`
            : res.response.primaryActivity;
        const reasons = [res.correctAnswer.primaryActivityReasoning, res.correctAnswer.interactingReasoning].filter(Boolean);
        return `
            <div class="feedback-item ${res.correct ? '' : 'wrong'}">
                ${videoHTML(trainingVideoURL(res.videoFilename), res.videoFilename, { autoplay: false })}
                <div>
                    <h3>${res.correct ? '✅' : '❌'} Practice clip ${idx + 1}</h3>
                    <table class="feedback-table">
                        <thead><tr><th>Question</th><th>Your answer</th><th>Correct answer</th></tr></thead>
                        <tbody>
                            <tr><td>Primary activity</td>${cell(res.primaryCorrect, yours)}<td>${escapeHTML(res.correctAnswer.primaryActivity)}</td></tr>
                            <tr><td>Anyone interacting</td>${cell(res.interactingCorrect, res.response.anyoneInteracting)}<td>${escapeHTML(res.correctAnswer.anyoneInteracting)}</td></tr>
                        </tbody>
                    </table>
                    ${!res.primaryCorrect ? `<p class="muted" style="font-size: 13px; margin-top: 8px;"><strong>${escapeHTML(res.correctAnswer.primaryActivity)}:</strong> ${escapeHTML(definitionOf(res.correctAnswer.primaryActivity))}</p>` : ''}
                    ${reasons.map(t => `<div class="reasoning">${escapeHTML(t)}</div>`).join('')}
                </div>
            </div>`;
    }).join('');

    let message;
    let buttons;
    if (result.passed) {
        message = '<p>You passed the training. Review the answers below, then continue to the main task.</p>';
        buttons = [['Begin main task →', 'main', true]];
    } else if (remaining > 0) {
        message = `<p>You need at least ${pct(training.passThreshold)} to pass. Review the correct answers below;
                   you have ${remaining} attempt${remaining === 1 ? '' : 's'} left with a new set of clips.</p>`;
        buttons = [['Review examples', 'examples'], ['Try again with new clips →', 'practice', true]];
    } else {
        message = `<p>You have used all ${training.maxAttempts} attempts without passing. Please contact the researcher.</p>`;
        buttons = devMode ? [['Continue anyway (dev) →', 'main', true]] : [['Finish', 'blocked', true]];
    }

    const r = await showScreen(`
        <div class="card score-card ${result.passed ? 'passed' : 'failed'}">
            <h2>${result.passed ? 'Training passed' : 'Training not passed'}</h2>
            <div class="score">${result.nCorrect} / ${result.nTotal}</div>
            <p class="muted">${pct(result.score)} correct · attempt ${result.attemptNumber} of ${training.maxAttempts}${result.devMode ? ' · dev mode, not recorded' : ''}</p>
            ${message}
        </div>
        <div class="feedback-list">${items}</div>
        <div class="actions" style="margin: 24px 0;">
            ${buttons.map(([label, value, primary]) =>
                `<button type="button" class="btn ${primary ? 'btn-primary' : ''}" data-done="${value}">${label}</button>`).join('')}
        </div>`);
    return r?.jump || r;
}

async function blockedPhase() {
    setStep('Training');
    const { maxAttempts } = state.config.training;
    const buttons = state.config.devMode ? [['Continue anyway (dev) →', 'main', true]] : [];
    const r = await messageScreen('Training not completed',
        `<p>You have used all ${maxAttempts} practice attempts for this task without passing.</p>
         <p>Please contact the researcher.</p>`, buttons);
    return r?.jump || (r === 'main' ? 'main' : null);
}

async function mainPhase() {
    setStep('Main task');
    let data;
    try {
        data = await api(assignmentURL());
    } catch (error) {
        if (error.status === 403) return 'blocked';
        throw error;
    }
    const { videos } = data;
    const annotations = data.devMode
        ? (state.devAnnotations[state.taskType] ||= {})
        : data.annotations;
    if (data.devMode) toast(DEV_SANDBOX_NOTE);

    if (videos.length === 0) {
        const r = await messageScreen('No clips available',
            '<p>There are no clips left to assign for this task. Please contact the researcher.</p>', []);
        return r?.jump || null;
    }

    const firstIncomplete = () => videos.findIndex(v => !annotations[v]);
    let index = firstIncomplete();
    if (index === -1) return 'done';

    while (true) {
        const filename = videos[index];
        setStep(`Main task · Clip ${index + 1} of ${videos.length}`);
        const r = await annotationScreen({
            videoURL: mainVideoURL(filename),
            filename,
            existing: annotations[filename],
            isPractice: false,
            index,
            total: videos.length,
            completedCount: Object.keys(annotations).length,
            canGoBack: index > 0
        });

        if (r?.jump) return r.jump;
        if (r.action === 'quit') return 'done';
        if (r.action === 'shuffle') {
            newPreviewSeed();
            state.devAnnotations[state.taskType] = {};
            toast('New preview clips');
            return 'main';
        }
        if (r.action === 'back') { index--; continue; }
        if (r.action === 'goto') { index = r.index; continue; }

        try {
            await api('/annotations', {
                method: 'POST',
                body: { taskType: state.taskType, videoFilename: filename, previewSeed: getPreviewSeed(), ...r.values }
            });
            annotations[filename] = r.values;
            toast(data.devMode ? '✓ Checked (dev mode: not saved)' : '✓ Saved');
        } catch (error) {
            toast('Failed to save: ' + error.message, true);
            continue;
        }

        if (index + 1 < videos.length) {
            index++;
        } else if (firstIncomplete() !== -1) {
            const left = videos.filter(v => !annotations[v]).length;
            toast(`${left} clip${left === 1 ? '' : 's'} still need annotating`);
            index = firstIncomplete();
        } else {
            return 'done';
        }
    }
}

async function donePhase() {
    setStep('Finished');
    const data = await api(assignmentURL());
    const { videos } = data;
    const annotations = data.devMode ? (state.devAnnotations[state.taskType] || {}) : data.annotations;
    const completed = Object.keys(annotations).length;
    const allDone = completed === videos.length;

    const buttons = [['Export my annotations (CSV)', 'export']];
    if (!allDone) buttons.push(['Return to clips', 'main', true]);

    while (true) {
        const r = await messageScreen(allDone ? '🎉 All clips annotated' : 'Progress saved',
            `<p style="font-size: 18px;">You have annotated <strong>${completed}</strong> of <strong>${videos.length}</strong> clips.</p>
             ${allDone ? '<p>Thank you! You can close this page.</p>'
                       : '<p>You can come back later and continue with the same ID.</p>'}`, buttons);
        if (r?.jump) return r.jump;
        if (r === 'main') return 'main';
        if (r === 'export') await exportAnnotations();
    }
}

const PHASES = {
    examples: examplesPhase,
    practice: practicePhase,
    main: mainPhase,
    blocked: blockedPhase,
    done: donePhase
};

async function exportAnnotations() {
    try {
        const response = await api('/export', { raw: true });
        const disposition = response.headers.get('content-disposition') || '';
        const filename = (disposition.match(/filename="?([^";]+)"?/) || [])[1] || 'annotations.csv';
        const url = URL.createObjectURL(await response.blob());
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
    } catch (error) {
        toast('Export failed: ' + error.message, true);
    }
}

// ============================================================================
// START
// ============================================================================

async function chooseTask() {
    const r = await showScreen(`
        <div class="card">
            <h2>Choose a task</h2>
            <div class="choice-grid">
                <button type="button" class="choice-card" data-done="do">
                    <strong>Doing</strong>${escapeHTML(state.config.activities.taskDescriptions.do)}
                </button>
                <button type="button" class="choice-card" data-done="see">
                    <strong>Seeing</strong>${escapeHTML(state.config.activities.taskDescriptions.see)}
                </button>
            </div>
        </div>`);
    history.replaceState(null, '', urlWithType(r));
    return r;
}

async function askAnnotatorId() {
    let lastId = '';
    try { lastId = localStorage.getItem(ID_STORAGE_KEY) || ''; } catch (e) { /* storage unavailable */ }

    return showScreen(`
        <form class="card" id="id-form" style="max-width: 460px;">
            <h2>Activity Annotations</h2>
            <p class="muted">${escapeHTML(state.config.activities.taskDescriptions[state.taskType])}</p>
            <label>Your annotator ID
                <input type="text" id="id-input" required autocomplete="off" value="${escapeHTML(lastId)}">
            </label>
            <p class="muted" style="font-size: 13px;">Use the same ID every time so your progress is restored.</p>
            <div class="actions"><button type="submit" class="btn btn-primary">Continue</button></div>
        </form>`, (root, done) => {
        const input = $('#id-input', root);
        input.focus();
        $('#id-form', root).addEventListener('submit', (e) => {
            e.preventDefault();
            const id = input.value.trim();
            if (id) done(id);
        });
    });
}

async function consent() {
    const r = await messageScreen('Consent agreement', `
        <div style="text-align: left;">
            <p><strong>By participating in this task, you agree that you will:</strong></p>
            <ul>
                <li>NOT attempt to reidentify individuals who appear in the stimuli</li>
                <li>NOT take screenshots of the stimuli</li>
                <li>NOT redistribute the stimuli in any form</li>
            </ul>
            <p>By clicking "I agree" below, you confirm that you understand and agree to these terms.</p>
        </div>`, [['I do not agree', 'no'], ['I agree', 'yes', true]]);
    return r === 'yes';
}

async function startingPhase() {
    const status = await api(`/training/status?taskType=${state.taskType}`);
    if (status.passed) {
        toast('Training already completed for this task');
        return 'main';
    }
    if (status.attemptsRemaining === 0) return 'blocked';
    return status.attemptsUsed > 0 ? 'practice' : 'examples';
}

async function main() {
    state.config = await api('/config');
    setupChrome();
    // With accounts, the username is the coder ID; otherwise the coder types an ID
    if (state.config.authMode === 'accounts') {
        const { user } = await api('/auth/me');
        state.annotator = (user || await showLogin()).username;
        $('#logout-btn').hidden = false;
    }

    // Dev mode can depend on who is logged in, so ask the server now
    const me = await api('/auth/me');
    state.config.devMode = me.devMode;
    showModeBadge();
    if (DEV_REQUESTED && !me.devMode) toast('Dev mode is only available to admin accounts', true);

    const typeParam = new URLSearchParams(location.search).get('type');
    state.taskType = ['do', 'see'].includes(typeParam) ? typeParam : await chooseTask();
    showTaskBadges();

    if (!state.annotator) {
        state.annotator = await askAnnotatorId();
        try { localStorage.setItem(ID_STORAGE_KEY, state.annotator); } catch (e) { /* storage unavailable */ }
    }
    await api('/login', { method: 'POST' });
    showTaskBadges();

    if (!(await consent())) {
        await messageScreen('Thank you', '<p>You must agree to the terms to take part. You can close this page.</p>', []);
        return;
    }

    $('#dev-nav').hidden = !state.config.devMode;
    let phase = await startingPhase();
    while (phase) phase = await PHASES[phase]();
}

main().catch(error => {
    console.error(error);
    $('#app').innerHTML = `<div class="card center"><h2>Something went wrong</h2>
        <p class="error">${escapeHTML(error.message)}</p><p>Please reload the page.</p></div>`;
});
