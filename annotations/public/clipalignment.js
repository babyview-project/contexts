// clipalignment.js - jsPsych Annotation Experiment
// Supports both Utterances Mode (4 utterances → 1 image) and Images Mode (4 images → 1 utterance)
// Pulls data from API with Prolific integration

// Configuration
const CONFIG = {
    API_BASE: 'http://localhost:3000/api'
};

// Global state
let credentials = { username: '', password: '' };
let prolificPid = '';
let studyId = '';
let sessionId = '';
let annotatorIndex = null;
let annotationsData = [];
let currentMode = '';
let currentIndex = 0;
let results = [];
let currentShuffledOrder = [];
let selectedOption = null;

let ASPECT_RATIO = "512 / 910"; // Width / Height
// Get URL parameters
function getUrlParams() {
    const params = new URLSearchParams(window.location.search);
    return {
        prolificPid: params.get('PROLIFIC_PID') || params.get('prolific_pid'),
        studyId: params.get('STUDY_ID') || params.get('study_id'),
        sessionId: params.get('SESSION_ID') || params.get('session_id')
    };
}

// API Helper Functions
function getAuthHeader() {
    return 'Basic ' + btoa(credentials.username + ':' + credentials.password);
}

async function apiCall(endpoint, options = {}) {
    const defaultOptions = {
        credentials: 'include',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': getAuthHeader()
        }
    };
    
    try {
        const response = await fetch(CONFIG.API_BASE + endpoint, {
            ...defaultOptions,
            ...options,
            headers: { ...defaultOptions.headers, ...options.headers }
        });
        
        if (!response.ok) {
            if (response.status === 401) {
                throw new Error('Authentication failed. Please check your username and password.');
            }
            const error = await response.json().catch(() => ({ error: 'Request failed' }));
            throw new Error(error.error || 'API request failed');
        }
        
        return response.json();
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

async function registerProlificUser(prolificPid, studyId, sessionId) {
    return apiCall('/clip-alignment/register', {
        method: 'POST',
        body: JSON.stringify({ prolificPid, studyId, sessionId })
    });
}

async function loadAnnotations(annotatorIndex) {
    return apiCall(`/clip-alignment/load?annotator_index=${annotatorIndex}`);
}

async function saveResults(resultsData) {
    return apiCall('/clip-alignment/results', {
        method: 'POST',
        body: JSON.stringify(resultsData)
    });
}

async function exportResults() {
    try {
        const response = await fetch(CONFIG.API_BASE + '/clip-alignment/export', {
            method: 'GET',
            credentials: 'include',
            headers: {
                'Authorization': getAuthHeader()
            }
        });
        
        if (!response.ok) {
            throw new Error('Export failed');
        }
        
        const contentDisposition = response.headers.get('content-disposition');
        let filename = `annotation_results_${currentMode}_${Date.now()}.csv`;
        if (contentDisposition) {
            const match = contentDisposition.match(/filename="?([^";]+)"?/);
            if (match) filename = match[1];
        }
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        
        console.log('Export successful');
    } catch (error) {
        alert('Export failed: ' + error.message);
        console.error('Export error:', error);
    }
}

// Utility Functions
function shuffleArray(array) {
    const shuffled = [...array];
    for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled;
}

function getProgressBarHTML() {
    const completed = results.length;
    const total = annotationsData.length;
    const progress = (completed / total) * 100;
    
    return `
        <div style="background: #f5f5f5; padding: 15px; margin-bottom: 20px; border-radius: 8px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <strong data-progress-completed>Completed: ${completed} / ${total}</strong>
                </div>
            </div>
            <div style="background: #ddd; border-radius: 10px; overflow: hidden; height: 30px; position: relative;">
                <div style="width: ${progress}%; height: 100%; background: linear-gradient(90deg, #4CAF50, #8BC34A); transition: width 0.3s;" id="progress-bar-fill"></div>
                <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-weight: bold; color: #333; z-index: 10;">
                    ${Math.round(progress)}%
                </div>
            </div>
        </div>
    `;
}

// Initialize jsPsych
const jsPsych = initJsPsych({
    display_element: 'jspsych-target',
    on_finish: function() {
        console.log('Experiment finished');
    }
});

// Timeline
let timeline = [];

// Get Prolific parameters and register user
const prolificRegistration = {
    type: jsPsychCallFunction,
    async: true,
    func: function(done) {
        const params = getUrlParams();
        
        if (!params.prolificPid) {
            alert('Error: Missing PROLIFIC_PID parameter. This experiment must be accessed through Prolific.');
            jsPsych.endExperiment('Missing Prolific parameters');
            done({ success: false, error: 'Missing PROLIFIC_PID' });
            return;
        }
        
        prolificPid = params.prolificPid;
        studyId = params.studyId || 'unknown';
        sessionId = params.sessionId || 'unknown';
        
        console.log('Prolific Params:', { prolificPid, studyId, sessionId });
        
        registerProlificUser(prolificPid, studyId, sessionId)
            .then((response) => {
                annotatorIndex = response.annotatorIndex;
                currentMode = response.mode;
                console.log('Registered:', { annotatorIndex, mode: currentMode });
                done({ success: true, annotatorIndex, mode: currentMode });
            })
            .catch((error) => {
                alert('Registration failed: ' + error.message);
                jsPsych.endExperiment('Registration failed');
                done({ success: false, error: error.message });
            });
    }
};
timeline.push(prolificRegistration);

// Welcome screen showing assigned mode
const welcomeScreen = {
    type: jsPsychHtmlButtonResponse,
    stimulus: function() {
        const modeDisplay = currentMode === 'utterances' ? 'Utterances Mode' : 'Images Mode';
        const modeColor = currentMode === 'utterances' ? '#2196F3' : '#4CAF50';
        const modeDescription = currentMode === 'utterances' 
            ? 'You will view one image and select the correct utterance from 4 options'
            : 'You will view one utterance and select the correct image from 4 options';
        
        return `
            <div style="max-width: 700px; margin: auto; padding: 40px; background: white; border-radius: 12px;">
                <h2 style="text-align: center; margin-bottom: 30px;">Welcome to the Annotation Study</h2>
                <div style="text-align: center; padding: 30px; background: ${modeColor}20; border-radius: 12px; border: 3px solid ${modeColor}; margin-bottom: 30px;">
                    <h3 style="color: ${modeColor}; margin-bottom: 20px;">Your Mode: ${modeDisplay}</h3>
                    <p style="font-size: 18px; line-height: 1.6; color: #555;">
                        ${modeDescription}
                    </p>
                </div>
                <div style="background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                    <p style="font-size: 16px; line-height: 1.8; color: #333;">
                        <strong>Instructions:</strong><br>
                        • Complete all annotations to the best of your ability<br>
                        • Your progress will be saved automatically<br>
                        • Click "Continue" when you're ready to begin
                    </p>
                </div>
            </div>
        `;
    },
    choices: ['Continue'],
    button_html: '<button class="jspsych-btn" style="font-size: 18px; padding: 15px 50px; margin: 20px;">%choice%</button>'
};
timeline.push(welcomeScreen);

// Load Annotations from API
const loadAnnotationsScreen = {
    type: jsPsychCallFunction,
    async: true,
    func: function(done) {
        loadAnnotations(annotatorIndex)
            .then((data) => {
                annotationsData = data.annotations;
                console.log('Loaded annotations:', annotationsData.length, 'for index:', annotatorIndex);
                done({ success: true, count: annotationsData.length });
            })
            .catch((error) => {
                alert('Failed to load annotations: ' + error.message);
                jsPsych.endExperiment('Failed to load data');
                done({ success: false, error: error.message });
            });
    }
};
timeline.push(loadAnnotationsScreen);

// Build annotation timeline
const buildAnnotationTimeline = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        return `
            <div style="text-align: center; margin: 100px auto;">
                <h2>Building experiment timeline...</h2>
                <p>Mode: ${currentMode === 'utterances' ? 'Utterances' : 'Images'}</p>
                <p>Preparing ${annotationsData.length} annotations</p>
            </div>
        `;
    },
    choices: "NO_KEYS",
    trial_duration: 500,
    on_load: function() {
        for (let i = 0; i < annotationsData.length; i++) {
            const annotationTrial = createAnnotationTrial(i);
            jsPsych.addNodeToEndOfTimeline(annotationTrial);
        }
        
        jsPsych.addNodeToEndOfTimeline(finalScreen);
    }
};
timeline.push(buildAnnotationTimeline);

// Create annotation trial based on mode
function createAnnotationTrial(index) {
    const annotation = annotationsData[index];
    if (currentMode === 'utterances') {
        return createUtterancesTrial(index, annotation);
    } else {
        return createImagesTrial(index, annotation);
    }
}

// Utterances Mode: Show 1 image, select from 4 utterances
function createUtterancesTrial(index, annotation) {
    // Shuffle utterances
    const utterances = [
        { text: annotation.utterance, isCorrect: true },
        { text: annotation.distractorUtt1, isCorrect: false },
        { text: annotation.distractorUtt2, isCorrect: false },
        { text: annotation.distractorUtt3, isCorrect: false }
    ];
    
    const shuffled = shuffleArray(utterances);
    const shuffledOrder = shuffled.map((u, idx) => ({
        position: idx,
        isCorrect: u.isCorrect,
        text: u.text
    }));
    
    const correctPosition = shuffledOrder.findIndex(u => u.isCorrect);
    
    return {
        type: jsPsychSurvey,
        survey_json: {
            showQuestionNumbers: "off",
            showCompleteButton: false,
            elements: [
                {
                    type: "html",
                    name: "progress_bar",
                    html: function() { return getProgressBarHTML(); }
                },
                /*
                {
                    type: "html",
                    name: "mode_indicator",
                    html: `
                        <div style="text-align: center; margin-bottom: 20px; padding: 10px; background: #e7f3ff; border-radius: 6px;">
                            <strong style="color: #1976D2; font-size: 16px;">Mode: Select Utterance for Image</strong>
                        </div>
                    `
                },
                */
                {
                    type: "html",
                    name: "image_display",
                    html: `
                        <div style="text-align: center; margin-bottom: 30px;">
                            <img src="${CONFIG.API_BASE}/clip-alignment/images/${annotation.imagePath}" 
                                 style="max-width: 100%; height: auto; max-height: 70vh; aspect-ratio: ${ASPECT_RATIO}; border-radius: 8px; border: 2px solid #ddd; object-fit: contain;"
                                 alt="Target Image">
                        </div>
                        <h4 style="text-align: center; margin-bottom: 20px;">Select the correct utterance:</h4>
                    `
                },
                {
                    type: "radiogroup",
                    name: "selected_utterance",
                    title: " ",
                    isRequired: true,
                    choices: shuffled.map((u, idx) => ({
                        value: idx,
                        text: u.text
                    })),
                    showLabel: false,
                    colCount: 2
                }
            ]
        },
        on_load: function() {
            currentIndex = index;
            selectedOption = null;
            currentShuffledOrder = shuffledOrder;
            
            setTimeout(() => {
                setupNavigationHandlers(index, correctPosition);
                
                // Add custom CSS for utterance tiles
                const style = document.createElement('style');
                style.textContent = `
                    /* Hide the Complete button */
                    .sd-action-bar, .sd-footer, .sd-completebtn, input[type="button"][value="Complete"] {
                        display: none !important;
                    }
                    
                    /* Make entire tile clickable and hide radio buttons */
                    .sd-selectbase__item {
                        margin: 10px !important;
                        padding: 20px !important;
                        background: #f8f9fa !important;
                        border: 2px solid #ddd !important;
                        border-radius: 8px !important;
                        cursor: pointer !important;
                        transition: all 0.2s ease !important;
                        position: relative !important;
                    }
                    .sd-selectbase__item:hover {
                        transform: scale(1.01);
                        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                        border-color: #bbb !important;
                    }
                    .sd-selectbase__item--checked {
                        background: #e3f2fd !important;
                        border: 3px solid #2196F3 !important;
                        box-shadow: 0 0 0 3px rgba(33, 150, 243, 0.2);
                    }
                    
                    /* Hide the actual radio button */
                    .sd-selectbase__item input[type="radio"],
                    .sd-item__control input[type="radio"],
                    .sd-radio__control {
                        display: none !important;
                        opacity: 0 !important;
                        position: absolute !important;
                        pointer-events: none !important;
                    }
                    
                    /* Make label take full space and be clickable */
                    .sd-selectbase__label,
                    .sd-item__control-label,
                    .sd-selectbase__control-label {
                        font-size: 26px !important;
                        line-height: 1.6 !important;
                        color: #333 !important;
                        cursor: pointer !important;
                        padding: 10px !important;
                        margin: 0 !important;
                        width: 100% !important;
                        display: block !important;
                        text-align: center !important;
                    }
                    
                    /* Remove any decorators or markers */
                    .sd-item__decorator,
                    .sd-selectbase__decorator {
                        display: none !important;
                    }
                `;
                document.head.appendChild(style);
                
                const root = document.querySelector('.jspsych-content') || document.getElementById('jspsych-target') || document.body;
                let radios = root.querySelectorAll('input[type="radio"][name="selected_utterance"]');

                if (!radios || radios.length === 0) {
                    // fallback: any radios on the page (best-effort)
                    radios = root.querySelectorAll('input[type="radio"]');
                }

                // If there are no radios, bail gracefully (so nothing throws).
                if (!radios || radios.length === 0) {
                    console.warn('Auto-advance: no radio inputs found for selected_utterance.');
                    return;
                }

                // Make entire tile clickable
                radios.forEach(radio => {
                    const tile = radio.closest('.sd-selectbase__item');
                    if (tile) {
                        tile.style.cursor = 'pointer';
                        tile.addEventListener('click', function(e) {
                            // Only trigger if not already clicking the radio directly
                            if (e.target !== radio) {
                                radio.click();
                            }
                        });
                    }
                });

                // Helper to update checked tile visuals
                function updateTileVisuals() {
                    radios.forEach(r => {
                        const tile = r.closest('.sd-selectbase__item');
                        if (tile) {
                            if (r.checked) tile.classList.add('sd-selectbase__item--checked');
                            else tile.classList.remove('sd-selectbase__item--checked');
                        }
                    });
                }

                // Attach change listeners to each radio input
                radios.forEach(radio => {
                    radio.addEventListener('change', (e) => {
                        // update tile visuals immediately
                        updateTileVisuals();

                        // read selected value (string) and convert to number if you prefer
                        const selectedValue = e.target.value;

                        // small delay so user sees the selection, then finish the trial
                        setTimeout(async () => {
                            // assemble any data you want to save for this trial
                            const trialData = {
                                selected_utterance: selectedValue,
                                imagePath: (annotation && annotation.imagePath) ? annotation.imagePath : null,
                                index: (typeof index !== 'undefined') ? index : null
                            };

                            // Advance the trial. jsPsych.finishTrial should be available in most setups.
                            if (typeof jsPsych !== 'undefined' && typeof jsPsych.finishTrial === 'function') {
                                const selectedPosition = parseInt(selectedValue);
                                const isCorrect = shuffledOrder[selectedPosition].isCorrect;
                                
                                const result = {
                                    row_index: index,
                                    prolific_pid: prolificPid,
                                    annotator_index: annotatorIndex,
                                    mode: currentMode,
                                    selected_position: selectedPosition,
                                    correct_position: correctPosition,
                                    is_correct: isCorrect,
                                    timestamp: new Date().toISOString(),
                                    ...annotation
                                };
                                
                                results.push(result);
                            
                                try {
                                    await saveResults({ results: [result] });
                                    console.log('Saved result for row:', index);
                                } catch (error) {
                                    console.error('Failed to save result:', error);
                                }
                                
                                jsPsych.finishTrial(trialData);
                            } else {
                                console.error('Auto-advance: could not find jsPsych.finishTrial to advance the trial.');
                            }
                        }, 300);
                    });
                });

                // In case a radio is pre-selected (e.g., from restore), sync visuals now
                updateTileVisuals();
            }, 100);
        }
    };
}

// Images Mode: Show 1 utterance, select from 4 images
function createImagesTrial(index, annotation) {
    // Shuffle images
    const images = [
        { path: annotation.imagePath, isCorrect: true },
        { path: annotation.distractorImg1, isCorrect: false },
        { path: annotation.distractorImg2, isCorrect: false },
        { path: annotation.distractorImg3, isCorrect: false }
    ];
    
    const shuffled = shuffleArray(images);
    const shuffledOrder = shuffled.map((img, idx) => ({
        position: idx,
        isCorrect: img.isCorrect,
        path: img.path
    }));
    
    const correctPosition = shuffledOrder.findIndex(img => img.isCorrect);
    
    return {
        type: jsPsychSurvey,
        survey_json: {
            showQuestionNumbers: "off",
            showCompleteButton: false,
            elements: [
                {
                    type: "html",
                    name: "progress_bar",
                    html: function() { return getProgressBarHTML(); }
                },
                /*
                {
                    type: "html",
                    name: "mode_indicator",
                    html: `
                        <div style="text-align: center; margin-bottom: 20px; padding: 10px; background: #e8f5e9; border-radius: 6px;">
                            <strong style="color: #388E3C; font-size: 16px;">Mode: Select Image for Utterance</strong>
                        </div>
                    `
                },
                */
                {
                    type: "html",
                    name: "utterance_display",
                    html: `
                        <div style="background: #f8f9fa; padding: 10px; border-radius: 8px;">
                            <h4 style="text-align: center;margin: 0;">Utterance:</h4>
                            <div style="font-size: 26px; line-height: 1.6; text-align: center;">
                                ${annotation.utterance}
                            </div>
                        </div>
                        <h4 style="text-align: center; margin-bottom: 20px;">Select the correct image:</h4>
                    `
                },
                {
                    type: "imagepicker",
                    name: "selected_image",
                    title: " ",
                    isRequired: true,
                    choices: shuffled.map((img, idx) => ({
                        value: idx,
                        imageLink: `${CONFIG.API_BASE}/clip-alignment/images/${img.path}`,
                        text: " "
                    })),
                    showLabel: true,
                    multiSelect: false,
                    imageFit: "contain",
                    imageHeight: 600,
                    imageWidth: 337,
                    colCount: 2
                }
            ]
        },
        on_load: function() {
            currentIndex = index;
            selectedOption = null;
            currentShuffledOrder = shuffledOrder;
            
            setTimeout(() => {
                setupNavigationHandlers(index, correctPosition);
                
                // Add custom CSS for aspect ratio
                const style = document.createElement('style');
                style.textContent = `
                    /* Hide the Complete button */
                    .sd-action-bar, .sd-footer, .sd-completebtn, input[type="button"][value="Complete"] {
                        display: none !important;
                    }
                    
                    .sd-imagepicker__item-decorator img {
                        aspect-ratio: ${ASPECT_RATIO} !important;
                        object-fit: contain !important;
                    }
                `;
                document.head.appendChild(style);
                
                const root = document.querySelector('.jspsych-content') || document.getElementById('jspsych-target') || document.body;
                let radios = root.querySelectorAll('input[type="radio"][name="selected_image"]');

                if (!radios || radios.length === 0) {
                    // fallback: any radios on the page (best-effort)
                    radios = root.querySelectorAll('input[type="radio"]');
                }

                // If there are no radios, bail gracefully (so nothing throws).
                if (!radios || radios.length === 0) {
                    console.warn('Auto-advance: no radio inputs found for selected_image.');
                    return;
                }

                // Make entire tile clickable
                radios.forEach(radio => {
                    const tile = radio.closest('.sd-imagepicker__item');
                    if (tile) {
                        tile.style.cursor = 'pointer';
                        tile.addEventListener('click', function(e) {
                            // Only trigger if not already clicking the radio directly
                            if (e.target !== radio) {
                                radio.click();
                            }
                        });
                    }
                });

                // Helper to update checked tile visuals
                function updateTileVisuals() {
                    radios.forEach(r => {
                        const tile = r.closest('.sd-imagepicker__item');
                        if (tile) {
                            if (r.checked) tile.classList.add('sd-imagepicker__item--checked');
                            else tile.classList.remove('sd-imagepicker__item--checked');
                        }
                    });
                }

                // Attach change listeners to each radio input
                radios.forEach(radio => {
                    radio.addEventListener('change', (e) => {
                        // update tile visuals immediately
                        updateTileVisuals();

                        // read selected value (string) and convert to number if you prefer
                        const selectedValue = e.target.value;

                        // small delay so user sees the selection, then finish the trial
                        setTimeout(async () => {
                            // assemble any data you want to save for this trial
                            const trialData = {
                                selected_image: selectedValue,
                                imagePath: (annotation && annotation.imagePath) ? annotation.imagePath : null,
                                index: (typeof index !== 'undefined') ? index : null
                            };

                            // Advance the trial. jsPsych.finishTrial should be available in most setups.
                            if (typeof jsPsych !== 'undefined' && typeof jsPsych.finishTrial === 'function') {
                                const selectedPosition = parseInt(selectedValue);
                                const isCorrect = shuffledOrder[selectedPosition].isCorrect;
                                
                                const result = {
                                    row_index: index,
                                    prolific_pid: prolificPid,
                                    annotator_index: annotatorIndex,
                                    mode: currentMode,
                                    selected_position: selectedPosition,
                                    correct_position: correctPosition,
                                    is_correct: isCorrect,
                                    timestamp: new Date().toISOString(),
                                    ...annotation
                                };
                                results.push(result);
                                
                                try {
                                    await saveResults({ results: [result] });
                                    console.log('Saved result for row:', index);
                                } catch (error) {
                                    console.error('Failed to save result:', error);
                                }
                                
                                jsPsych.finishTrial(trialData);
                            } else {
                                console.error('Auto-advance: could not find jsPsych.finishTrial to advance the trial.');
                            }
                        }, 300);
                    });
                });

                // In case a radio is pre-selected (e.g., from restore), sync visuals now
                updateTileVisuals();
                setTimeout(() => {
                    window.scrollTo({ top: 0, behavior: "smooth" });
                }, 50);
            }, 100);
        }
    };
}

// Setup navigation handlers for jump and quit buttons
function setupNavigationHandlers(currentIdx, correctPosition) {
    const jumpButton = document.getElementById('jump-button');
    const jumpInput = document.getElementById('jump-to-row');
    const quitButton = document.getElementById('quit-button');
    
    const handleJump = () => {
        const targetRowNum = parseInt(jumpInput.value);
        if (isNaN(targetRowNum) || targetRowNum < 1 || targetRowNum > annotationsData.length) {
            alert(`Please enter a valid row number between 1 and ${annotationsData.length}`);
            return;
        }
        
        const targetIndex = targetRowNum - 1;
        
        if (targetIndex !== currentIdx) {
            jsPsych.endCurrentTimeline();
            const newTimeline = [];
            
            for (let i = targetIndex; i < annotationsData.length; i++) {
                const trial = createAnnotationTrial(i);
                newTimeline.push(trial);
            }
            
            newTimeline.push(finalScreen);
            jsPsych.run(newTimeline);
        }
    };
    
    const handleQuit = () => {
        if (confirm('Are you sure you want to quit? Your progress has been saved.')) {
            jsPsych.endCurrentTimeline();
            jsPsych.run([finalScreen]);
        }
    };
    
    if (jumpButton) {
        jumpButton.addEventListener('click', handleJump);
    }
    
    if (quitButton) {
        quitButton.addEventListener('click', handleQuit);
    }
    
    if (jumpInput) {
        jumpInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                handleJump();
            }
        });
    }
}

// Final screen
const finalScreen = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        const completed = results.length;
        const total = annotationsData.length;
        const correctCount = results.filter(r => r.is_correct).length;
        const accuracy = total > 0 ? ((correctCount / completed) * 100).toFixed(1) : 0;
        
        return `
            <div style="max-width: 600px; margin: auto; padding: 40px; background: white; border-radius: 12px; text-align: center;">
                <h2>🎉 Annotation Complete!</h2>
                <div style="margin: 30px 0;">
                    <p style="font-size: 20px; margin: 10px 0;">
                        <strong>Completed:</strong> ${completed} / ${total} annotations
                    </p>
                    <p style="font-size: 20px; margin: 10px 0;">
                        <strong>Correct:</strong> ${correctCount} / ${completed}
                    </p>
                    <p style="font-size: 20px; margin: 10px 0;">
                        <strong>Accuracy:</strong> ${accuracy}%
                    </p>
                    <p style="font-size: 18px; margin: 10px 0; color: #666;">
                        <strong>Mode:</strong> ${currentMode === 'utterances' ? 'Utterances' : 'Images'}
                    </p>
                </div>
                <p style="margin: 20px 0;">Thank you for participating! You may now close this window or return to Prolific.</p>
            </div>
        `;
    },
    choices: "NO_KEYS",
    trial_duration: null,
    on_finish: function(data) {
        // Optionally redirect back to Prolific
        // window.location.href = 'https://app.prolific.co/submissions/complete?cc=COMPLETION_CODE';
    }
};


jsPsych.run(timeline);