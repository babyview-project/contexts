// experiment.js - jsPsych Activity Annotation Experiment
// Videos stored in browser memory only (blob URLs)
// Note: Include JSZip library in your HTML: <script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>

// Configuration
const CONFIG = {
    API_BASE: 'https://ucsdlearninglabs.org/bvannotations/api/'
};

// Global state
let credentials = { username: '', password: '' };
let annotatorName = '';
let videosData = [];
let videoBlobs = {}; // Store blob URLs for videos in browser memory
let currentVideoIndex = 0;
let dropdownOptions = {};
let completedVideos = new Set();

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

async function getAnnotationsForVideos(videoFilenames) {
    return apiCall('/get-annotations-for-videos', {
        method: 'POST',
        body: JSON.stringify({ videoFilenames })
    });
}

async function saveAnnotation(data) {
    console.log(data)
    return apiCall('/annotations', {
        method: 'POST',
        body: JSON.stringify(data)
    });
}

async function getOptions() {
    return apiCall('/options');
}

async function exportAnnotations() {
    try {
        const response = await fetch(CONFIG.API_BASE + '/export', {
            method: 'GET',
            credentials: 'include',
            headers: {
                'Authorization': getAuthHeader()
            }
        });
        
        if (!response.ok) {
            throw new Error('Export failed');
        }
        
        // Get filename from response headers or use default
        const contentDisposition = response.headers.get('content-disposition');
        let filename = 'annotations.csv';
        if (contentDisposition) {
            const match = contentDisposition.match(/filename="?([^";]+)"?/);
            if (match) filename = match[1];
        }
        
        // Download the file
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

function showSaveIndicator() {
    const indicator = document.getElementById('saveIndicator');
    if (indicator) {
        indicator.classList.add('show');
        setTimeout(() => {
            indicator.classList.remove('show');
        }, 2000);
    }
}

// Helper function to update progress bar dynamically
function updateProgressBar() {
    const completed = completedVideos.size;
    const total = videosData.length;
    const progress = (completed / total) * 100;
    
    const progressBarFill = document.getElementById('progress-bar-fill');
    const progressBarText = progressBarFill?.nextElementSibling;
    const completedText = document.querySelector('[data-progress-completed]');
    
    if (progressBarFill) {
        progressBarFill.style.width = progress + '%';
    }
    if (progressBarText) {
        progressBarText.textContent = Math.round(progress) + '%';
    }
    if (completedText) {
        completedText.textContent = `Completed: ${completed} / ${total}`;
        completedText.style.marginLeft = '5px';
    }
}

// Helper function to extract videos from zip file
async function extractVideosFromZip(zipFile) {
    if (typeof JSZip === 'undefined') {
        throw new Error('JSZip library not loaded. Please include it in your HTML.');
    }
    
    const zip = new JSZip();
    const zipData = await zip.loadAsync(zipFile);
    const videoFiles = [];
    
    // Extract video files
    const videoExtensions = ['.mp4', '.webm', '.ogg', '.mov', '.avi'];
    for (const [filename, file] of Object.entries(zipData.files)) {
        if (!file.dir && !filename.startsWith('__MACOSX/') && videoExtensions.some(ext => filename.toLowerCase().endsWith(ext))) {
            const blob = await file.async('blob');
            const videoFile = new File([blob], filename.split('/').pop(), { type: 'video/mp4' });
            videoFiles.push(videoFile);
        }
    }
    
    return videoFiles;
}

// Initialize jsPsych
const jsPsych = initJsPsych({
    display_element: 'jspsych-target',
    on_finish: function() {
        jsPsych.data.displayData();
    }
});

// Timeline
let timeline = [];

// Authentication Screen
let loginSuccess = false;

const authTrial = {
    type: jsPsychSurveyHtmlForm,
    preamble: '<h2 style="text-align: center; margin-bottom: 30px;">🔐 Login to Activity Annotation</h2>',
    html: `
      <label>Username:<br>
        <input name="username" type="text" required placeholder="Enter your username">
      </label><br><br>
  
      <label>Password:<br>
        <input name="password" type="password" required placeholder="Enter your password">
      </label>
      <br><br>
    `,
    button_label: 'Login',
    on_finish: function(data) {
      credentials.username = data.response.username;
      credentials.password = data.response.password;
    }
  };
  
  const checkAuth = {
    type: jsPsychCallFunction,
    async: true,
    func: function(done) {
      apiCall('/health')
        .then(() => {
          loginSuccess = true;
          done({ success: true });
        })
        .catch((err) => {
          alert("Authentication failed. Please try again.");
          loginSuccess = false;
          done({ success: false });
        });
    }
  };
  
  const loginLoop = {
    timeline: [authTrial, checkAuth],
    loop_function: function() {
      return !loginSuccess;
    }
  };

  //timeline.push(loginLoop);

// User Name Screen
const nameScreen = {
    type: jsPsychSurveyText,
    questions: [
        {
            prompt: '<h2>Activity Annotations</h2><p style="font-size: 20px;">Please enter your ID:</p>',
            name: 'annotator_name',
            required: true,
        }
    ],
    button_label: 'Continue',

    on_finish: async function(data) {
        annotatorName = data.response.annotator_name;
        try {
            await apiCall('/login', {
                method: 'POST',
                body: JSON.stringify({ name: annotatorName })
            });
            
            const optionsData = await getOptions();
            dropdownOptions = optionsData;
        } catch (error) {
            alert('Login failed: ' + error.message);
            //jsPsych.endExperiment('Login failed');
        }
    }
};
timeline.push(nameScreen);

// Global file storage (must be outside trial definition)
window._videoFiles = null;
let filesUploaded = false;
// Video ZIP Upload Screen
const fileUpload = {
    type: jsPsychHtmlButtonResponse,
    stimulus: `
        <div style="max-width: 700px; margin: auto; padding: 40px; background: white; border-radius: 12px;">            
            <h2 style="text-align: center; margin-bottom: 20px;">Upload videos ZIP</h2>
            
            <div style="background: #e3f2fd; padding: 15px; margin: 20px 0; border-radius: 8px; border-left: 4px solid #2196F3;">
                <strong>Instructions:</strong> Upload a ZIP file containing all your video files. Videos will be automatically detected and sorted alphabetically.
            </div>
            
            <div style="margin: 25px 0;">
                <label style="display: block; margin-bottom: 10px; font-weight: bold; font-size: 16px;">
                    Select ZIP File:
                </label>
                <input type="file" id="videoZip" accept=".zip"
                       style="width: 100%; padding: 15px; border: 2px dashed #2196F3; border-radius: 8px; background: #f5f5f5;">
                <div id="videoCount" style="margin-top: 10px; color: #666; font-size: 14px;">
                    No ZIP selected
                </div>
            </div>
        </div>
    `,
    choices: ['Continue'],
    button_html: '<button class="jspsych-btn" style="font-size: 18px; padding: 12px 30px;">%choice%</button>',
    on_load: function() {
        const zipInput = document.getElementById('videoZip');
        
        if (zipInput) {
            zipInput.addEventListener('change', async function(e) {
                const file = e.target.files[0];
                
                if (!file) return;
                
                if (!file.name.toLowerCase().endsWith('.zip')) {
                    alert('Please select a ZIP file');
                    return;
                }
                
                document.getElementById('videoCount').textContent = 'Extracting videos from ZIP...';
                document.getElementById('videoCount').style.color = '#FF9800';
                
                try {
                    const extractedVideos = await extractVideosFromZip(file);
                    
                    // Sort videos alphabetically by filename
                    extractedVideos.sort((a, b) => a.name.localeCompare(b.name));
                    
                    window._videoFiles = extractedVideos;
                    const count = extractedVideos.length;
                    
                    if (count > 0) {
                        const totalSize = extractedVideos.reduce((sum, f) => sum + f.size, 0);
                        const sizeMB = (totalSize / (1024 * 1024)).toFixed(2);
                        document.getElementById('videoCount').textContent = 
                            `✓ ${count} video(s) extracted (${sizeMB} MB)`;
                        document.getElementById('videoCount').style.color = '#4CAF50';
                        document.getElementById('videoCount').style.fontWeight = 'bold';
                    } else {
                        document.getElementById('videoCount').textContent = 
                            '⚠ No videos found in ZIP';
                        document.getElementById('videoCount').style.color = '#f44336';
                    }
                } catch (error) {
                    alert('Error extracting ZIP: ' + error.message);
                    console.error(error);
                    document.getElementById('videoCount').textContent = 
                        '✗ Error extracting ZIP';
                    document.getElementById('videoCount').style.color = '#f44336';
                }
            });
        }
    },
    on_finish: function(data) {
        // Store files for processing in next trial
        data.videoFiles = window._videoFiles;
    }
};

const processFiles = {
    type: jsPsychCallFunction,
    async: true,
    func: function(done) {
        const videoFiles = window._videoFiles;
        
        if (!videoFiles || videoFiles.length === 0) {
            alert('Please select a ZIP file with videos before continuing');
            return;
        }
        
        filesUploaded = true;
        
        // Create video blobs and build video data
        videoFiles.forEach((file, idx) => {
            const blobUrl = URL.createObjectURL(file);
            videoBlobs[file.name] = blobUrl;
            
            videosData.push({
                videoFilename: file.name,
                description: '',
                order: idx + 1
            });
        });
        
        // Get existing annotations for these videos
        const videoFilenames = videosData.map(v => v.videoFilename);
        
        getAnnotationsForVideos(videoFilenames)
            .then((result) => {
                const annotationMap = result.annotations;
                
                // Add existing annotations to video data
                videosData.forEach((video, idx) => {
                    if (annotationMap[video.videoFilename]) {
                        video.existingAnnotation = annotationMap[video.videoFilename];
                        completedVideos.add(idx);
                    }
                });
                
                // Find the last contiguous annotated video
                let lastAnnotatedIndex = -1;
                for (let i = 0; i < videosData.length; i++) {
                    if (completedVideos.has(i)) {
                        lastAnnotatedIndex = i;
                    } else {
                        break;
                    }
                }
                
                // Set current video to the one after the last annotated, or 0 if none annotated
                currentVideoIndex = lastAnnotatedIndex + 1;
                if (currentVideoIndex >= videosData.length) {
                    currentVideoIndex = 0;
                }
                
                console.log(`Loaded ${videosData.length} videos, starting at index ${currentVideoIndex}`);
                done({ success: true, videosData: videosData, startIndex: currentVideoIndex });
            })
            .catch((error) => {
                alert('Setup failed: ' + error.message);
                jsPsych.endExperiment('Setup failed');
                done({ success: false, error: error.message });
            });
    }
};

const fileUploadLoop = {
    timeline: [fileUpload, processFiles],
    loop_function: function() {
      return !filesUploaded;
    }
  };

timeline.push(fileUploadLoop);

// Final screen
const finalScreen = {
    type: jsPsychHtmlButtonResponse,
    stimulus: function() {
        const completed = completedVideos.size;
        const total = videosData.length;
        return `
            <div style="max-width: 600px; margin: auto; padding: 40px; background: white; border-radius: 12px; text-align: center;">
                <h2>🎉 Annotation Complete!</h2>
                <p style="font-size: 20px; margin: 20px 0;">
                    You completed <strong>${completed}</strong> out of <strong>${total}</strong> videos.
                </p>
                <p>Click the button below to export your annotations as CSV.</p>
            </div>
        `;
    },
    choices: ['Export Annotations', 'Finish'],
    on_finish: function(data) {
        if (data.response === 0) {
            exportAnnotations();
        }
    }
};

// Build video timeline dynamically
const buildVideoTimeline = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: function() {
        return `
            <div style="text-align: center; margin: 100px auto;">
                <h2>Building experiment timeline...</h2>
                <p>Preparing ${videosData.length} videos for annotation</p>
                <p>Starting at video ${currentVideoIndex + 1}</p>
            </div>
        `;
    },
    choices: "NO_KEYS",
    trial_duration: 500,
    on_load: function() {
        console.log(videosData.length)
        
        // Add videos starting from currentVideoIndex
        for (let i = currentVideoIndex; i < videosData.length; i++) {
            const videoTrialSet = createVideoTrial(i);
            jsPsych.addNodeToEndOfTimeline(videoTrialSet);
        }
        
        // Add remaining videos from the beginning if we didn't start at 0
        if (currentVideoIndex > 0) {
            for (let i = 0; i < currentVideoIndex; i++) {
                const videoTrialSet = createVideoTrial(i);
                jsPsych.addNodeToEndOfTimeline(videoTrialSet);
            }
        }
        
        jsPsych.addNodeToEndOfTimeline(finalScreen);
    }
};

timeline.push(buildVideoTimeline);

// Create video annotation trials
function createVideoTrial(videoIndex) {
    const video = videosData[videoIndex];
    const videoUrl = videoBlobs[video.videoFilename];
    
    // Survey trial with integrated layout
    const surveyTrial = {
        type: jsPsychSurvey,
        survey_json: {
            showQuestionNumbers: "off",
            elements: [
                {
                    type: "html",
                    name: "video_and_progress",
                    html: `
                        <style>
                            .video-annotation-container {
                                display: flex;
                                gap: 12px;
                                max-width: 100%;
                                padding: 10px;
                                height: 77vh;
                            }
                            .video-section {
                                flex: 0 0 60%;
                                display: flex;
                                flex-direction: column;
                                gap: 8px;
                            }
                            .questions-section {
                                flex: 0 0 40%;
                                padding: 10px;
                                background: #f5f5f5;
                                border-radius: 8px;
                                overflow-y: auto;
                                max-height: 75vh;
                                display: flex;
                                flex-direction: column;
                            }
                            .progress-bar-container {
                                background: white;
                                padding: 8px;
                                border-radius: 8px;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                            }
                            .video-player {
                                background: #000;
                                padding: 10px;
                                border-radius: 8px;
                                display: flex;
                                align-items: center;
                                justify-content: center;
                                max-height: 75vh;
                                aspect-ratio: 512 / 910;
                            }
                            .video-player video {
                                width: 100%;
                                max-height: 74vh;
                                aspect-ratio: 512 / 910;
                                object-fit: contain;
                            }
                            .progress-controls {
                                display: flex;
                                justify-content: space-between;
                                align-items: center;
                                margin-bottom: 10px;
                            }
                            .jump-controls {
                                display: flex;
                                align-items: center;
                                gap: 10px;
                            }
                            .jump-controls input {
                                width: 80px;
                                padding: 5px;
                                border: 2px solid #2196F3;
                                border-radius: 4px;
                                font-size: 14px;
                            }
                            .jump-controls button {
                                padding: 5px 15px;
                                border: none;
                                border-radius: 4px;
                                cursor: pointer;
                                font-size: 14px;
                                color: white !important;
                            }
                            .jump-button {
                                background: #2196F3 !important;
                            }
                            .jump-button:hover {
                                background: #1976D2 !important;
                            }
                            .quit-button {
                                background: #f44336 !important;
                            }
                            .quit-button:hover {
                                background: #d32f2f !important;
                            }
                            .progress-bar {
                                background: #ddd;
                                border-radius: 10px;
                                overflow: hidden;
                                height: 30px;
                                position: relative;
                            }
                            .progress-bar-fill {
                                height: 100%;
                                background: linear-gradient(90deg, #4CAF50, #8BC34A);
                                transition: width 0.3s;
                            }
                            .progress-text {
                                position: absolute;
                                top: 50%;
                                left: 50%;
                                transform: translate(-50%, -50%);
                                font-weight: bold;
                                color: #333;
                                z-index: 10;
                            }
                            .questions-section h3 {
                                margin-top: 0;
                                margin-bottom: 8px;
                                font-size: 16px;
                            }
                            .sd-question {
                                margin-bottom: 8px !important;
                                padding-bottom: 4px !important;
                            }
                            .sd-question__title {
                                font-size: 13px !important;
                                margin-bottom: 4px !important;
                                line-height: 1.3 !important;
                            }
                            .sd-input,
                            .sd-dropdown,
                            .sd-tagbox {
                                font-size: 13px !important;
                                padding: 4px 6px !important;
                            }
                            .sd-element {
                                padding: 10px !important;
                            }
                            .sd-row {
                                padding: 10px !important;
                            }
                            .sd-element--with-frame {
                                padding: 6px !important;
                                margin-bottom: 6px !important;
                            }
                            .sd-row__question {
                                padding: 10px !important;
                            }
                            .sd-row__question--small {
                                padding: 10px !important;
                            }
                            .jspsych-question-root {
                                padding: 10px !important;
                                margin: 0 !important;
                            }
                            /* Make submit button more compact */
                            .sd-action-bar {
                                padding-top: 8px !important;
                                margin-top: 8px !important;
                            }
                            .sd-btn {
                                padding: 8px 16px !important;
                                font-size: 14px !important;
                            }
                        </style>
                        
                        <div class="video-annotation-container">
                            <div class="video-section">
                                <div class="progress-bar-container">
                                    <div class="progress-controls">
                                        <strong>Video: ${videoIndex + 1} / ${videosData.length}</strong>
                                        <div class="jump-controls">
                                            <label style="margin: 0; font-size: 14px;">Jump to:</label>
                                            <input type="number" id="jump-to-video" min="1" max="${videosData.length}" 
                                                   placeholder="${videoIndex + 1}">
                                            <button class="jump-button" id="jump-button">Go</button>
                                            <button class="quit-button" id="quit-button">Quit</button>
                                        </div>
                                        <strong data-progress-completed>Completed: ${completedVideos.size} / ${videosData.length}</strong>
                                    </div>
                                    <div class="progress-bar">
                                        <div class="progress-bar-fill" id="progress-bar-fill" style="width: ${(completedVideos.size / videosData.length) * 100}%"></div>
                                        <div class="progress-text">${Math.round((completedVideos.size / videosData.length) * 100)}%</div>
                                    </div>
                                </div>
                                
                                <div class="video-player">
                                    <video controls autoplay>
                                        <source src="${videoUrl}" type="video/mp4">
                                        Your browser does not support the video tag.
                                    </video>
                                </div>
                            </div>
                            
                            <div class="questions-section" id="questions-container-${videoIndex}">
                                <h3 style="margin-top: 0;">Annotation Questions</h3>
                            </div>
                        </div>
                        
                        <div id="questions-placeholder"></div>
                    `
                },
                
                /* ───────────────────────────────
                   Questions (will appear in side panel)
                ─────────────────────────────── */
                {
                    type: "dropdown",
                    name: "primaryActivity",
                    title: "1. What is the primary activity that the child wearing the camera is doing?",
                    placeholder: "Select primary activity...",
                    isRequired: true,
                    choices: dropdownOptions.activities,   
                },
                {
                    type: "text",
                    name: "primaryActivityOther",
                    title: "Please specify the primary activity:",
                    isRequired: true,
                    visibleIf: "{primaryActivity} = 'other'"
                },
                {
                    type: "dropdown",
                    name: "primaryActivityConfidence",
                    title: "2. How confident are you?",
                    isRequired: true,
                    choices: [
                        { value: "1", text: "1 - Low" },
                        { value: "2", text: "2 - Medium" },
                        { value: "3", text: "3 - High" }
                    ]
                },
                {
                    type: "tagbox",
                    name: "otherActivities",
                    title: "3. What other activities is the child wearing the camera doing? [multi-select]",
                    placeholder: "Search and select activities...",
                    isRequired: true,
                    choices: [...dropdownOptions.activities, "none"]
                },
                {
                    type: "text",
                    name: "otherActivitiesOther",
                    title: "Please specify the other activities (comma-separated if multiple):",
                    isRequired: true,
                    visibleIf: "{otherActivities} contains 'other'"
                },
                {
                    type: "dropdown",
                    name: "otherActivitiesConfidence",
                    title: "4. How confident are you?",
                    isRequired: true,
                    choices: [
                        { value: "1", text: "1 - Low" },
                        { value: "2", text: "2 - Medium" },
                        { value: "3", text: "3 - High" }
                    ]
                },
                {
                    type: "dropdown",
                    name: "anyoneInteracting",
                    title: "5. Is anyone interacting with the child?",
                    isRequired: true,
                    choices: ["yes", "no"]
                }
            ]
        },
        survey_function: function(survey) {
            survey.data = video.existingAnnotation;
            survey.onValueChanged.add(function(sender, options) {
                if (options.name === "primaryActivity") {
                    const primaryActivityOther = sender.getQuestionByName("primaryActivityOther");
                    if (primaryActivityOther) {
                        primaryActivityOther.visible = (options.value === "other");
                    }
                }
                
                if (options.name === "otherActivities") {
                    const otherActivitiesOther = sender.getQuestionByName("otherActivitiesOther");
                    if (otherActivitiesOther) {
                        const hasOther = options.value && options.value.includes("other");
                        otherActivitiesOther.visible = hasOther;
                    }
                }
            });
            // Move questions into the side panel after render
            survey.onAfterRenderSurvey.add(function(sender) {
                setTimeout(() => {
                    const questionsSection = document.getElementById(`questions-container-${videoIndex}`);
                    const questionsPlaceholder = document.getElementById('questions-placeholder');
                    
                    if (!questionsSection) {
                        console.log('Could not find questions section');
                        return;
                    }
                    
                    // Get all question rows (skip the first HTML element)
                    const allRows = document.querySelectorAll('.sd-row');
                    console.log('Found rows:', allRows.length);
                    
                    // Move questions (all rows except first one which is video/progress)
                    allRows.forEach((row, index) => {
                        if (index > 0) { // Skip first row (video HTML)
                            questionsSection.appendChild(row);
                        }
                    });
                    
                    // Remove the placeholder
                    if (questionsPlaceholder && questionsPlaceholder.parentNode) {
                        questionsPlaceholder.parentNode.removeChild(questionsPlaceholder);
                    }
                }, 200);
            });
            
            // Handle dynamic question rendering (for conditional questions)
            survey.onAfterRenderQuestion.add(function(sender, options) {
                setTimeout(() => {
                    const questionsSection = document.getElementById(`questions-container-${videoIndex}`);
                    if (questionsSection && options.htmlElement) {
                        const row = options.htmlElement.closest('.sd-row');
                        if (row && row.parentNode !== questionsSection) {
                            questionsSection.appendChild(row);
                        }
                    }
                }, 50);
            });
        },
        on_load: function() {
            // Add handler for jump and quit buttons
            setTimeout(() => {
                const jumpButton = document.getElementById('jump-button');
                const jumpInput = document.getElementById('jump-to-video');
                const quitButton = document.getElementById('quit-button');
                
                const handleJump = () => {
                    const targetVideoNum = parseInt(jumpInput.value);
                    if (isNaN(targetVideoNum) || targetVideoNum < 1 || targetVideoNum > videosData.length) {
                        alert(`Please enter a valid video number between 1 and ${videosData.length}`);
                        return;
                    }
                    
                    const targetIndex = targetVideoNum - 1;
                    
                    if (targetIndex !== videoIndex) {
                        jsPsych.endCurrentTimeline();
                        new_timeline = []
                        for (let i = targetIndex; i < videosData.length; i++) {
                            const videoTrialSet = createVideoTrial(i);
                            jsPsych.addNodeToEndOfTimeline(videoTrialSet);
                            new_timeline.push(videoTrialSet)
                        }
                        new_timeline.push(finalScreen)
                        jsPsych.run(new_timeline)
                    }
                };
                
                const handleQuit = () => {
                    if (confirm('Are you sure you want to quit? Your progress has been saved.')) {
                        jsPsych.endCurrentTimeline();
                        jsPsych.run([finalScreen]);
                    }
                };
                
                if (jumpButton) {
                    jumpButton.removeEventListener('click', handleJump);
                    jumpButton.addEventListener('click', handleJump);
                    console.log('Jump button handler attached');
                }
                
                if (quitButton) {
                    quitButton.removeEventListener('click', handleQuit);
                    quitButton.addEventListener('click', handleQuit);
                    console.log('Quit button handler attached');
                }
                
                if (jumpInput) {
                    jumpInput.addEventListener('keypress', function(e) {
                        if (e.key === 'Enter') {
                            handleJump();
                        }
                    });
                }
            }, 500);
        },
        on_finish: async function(data) {            
            const trialData = data.response;
            
            const annotationData = {
                videoFilename: video.videoFilename,
                description: video.description,
                ...trialData
            };
            
            try {
                video.existingAnnotation = annotationData;
                await saveAnnotation(annotationData);
                completedVideos.add(videoIndex);
                showSaveIndicator();
                updateProgressBar();
                console.log('Saved:', video.videoFilename);
            } catch (error) {
                alert('Failed to save: ' + error.message);
            }
        }
    };

    return {
        timeline: [surveyTrial]
    };
}

// Run the experiment
jsPsych.run(timeline);