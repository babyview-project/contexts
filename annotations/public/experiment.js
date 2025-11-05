// experiment.js - jsPsych Activity Annotation Experiment
// Videos stored in browser memory only (blob URLs)
// Note: Include JSZip library in your HTML: <script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>

// Configuration
const CONFIG = {
    API_BASE: 'http://localhost:3000/api'
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

async function uploadCSV(file) {
    const formData = new FormData();
    formData.append('csvFile', file);
    
    try {
        const response = await fetch(CONFIG.API_BASE + '/upload-csv', {
            method: 'POST',
            body: formData,
            credentials: 'include',
            headers: {
                'Authorization': getAuthHeader()
            }
        });
        
        if (!response.ok) {
            if (response.status === 401) {
                throw new Error('Authentication failed.');
            }
            const error = await response.json().catch(() => ({ error: 'Upload failed' }));
            throw new Error(error.error || 'Upload failed');
        }
        
        return response.json();
    } catch (error) {
        console.error('Upload Error:', error);
        throw error;
    }
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
window._csvFile = null;
let filesUploaded = false;
// Video and CSV Upload Screen
const fileUpload = {
    type: jsPsychHtmlButtonResponse,
    stimulus: `
        <div style="max-width: 700px; margin: auto; padding: 40px; background: white; border-radius: 12px;">            
            <div style="background: #e3f2fd; padding: 15px; margin: 20px 0; border-radius: 8px; border-left: 4px solid #2196F3;">
                <strong>Instructions:</strong>
                CSV must contain: <code>video_filename</code>, <code>description</code>, optionally <code>order</code>. The <code>video_filename</code> in CSV should match your video filenames exactly including extension
            </div>
            
            <div style="margin: 25px 0;">
                <label style="display: block; margin-bottom: 10px; font-weight: bold; font-size: 16px;">
                    Select Video Files (or ZIP containing videos):
                </label>
                <input type="file" id="videoFiles" accept="video/*,.zip" multiple
                       style="width: 100%; padding: 15px; border: 2px dashed #2196F3; border-radius: 8px; background: #f5f5f5;">
                <div id="videoCount" style="margin-top: 10px; color: #666; font-size: 14px;">
                    No videos selected
                </div>
            </div>
            
            <div style="margin: 25px 0;">
                <label style="display: block; margin-bottom: 10px; font-weight: bold; font-size: 16px;">
                    Select CSV File:
                </label>
                <input type="file" id="csvFile" accept=".csv"
                       style="width: 100%; padding: 15px; border: 2px dashed #4CAF50; border-radius: 8px; background: #f5f5f5;">
                <div id="csvIndicator" style="margin-top: 10px; color: #666; font-size: 14px;">
                    No CSV selected
                </div>
            </div>
        </div>
    `,
    choices: ['Continue'],
    button_html: '<button class="jspsych-btn" style="font-size: 18px; padding: 12px 30px;">%choice%</button>',
    on_load: function() {
        const videoInput = document.getElementById('videoFiles');
        const csvInput = document.getElementById('csvFile');
        
        if (videoInput) {
            videoInput.addEventListener('change', async function(e) {
                const files = e.target.files;
                let allVideoFiles = [];
                
                // Check if any file is a zip
                for (let i = 0; i < files.length; i++) {
                    const file = files[i];
                    if (file.name.toLowerCase().endsWith('.zip')) {
                        document.getElementById('videoCount').textContent = 'Extracting videos from ZIP...';
                        document.getElementById('videoCount').style.color = '#FF9800';
                        try {
                            const extractedVideos = await extractVideosFromZip(file);
                            allVideoFiles = allVideoFiles.concat(extractedVideos);
                        } catch (error) {
                            alert('Error extracting ZIP: ' + error.message);
                            console.error(error);
                        }
                    } else if (file.type.startsWith('video/')) {
                        allVideoFiles.push(file);
                    }
                }
                
                window._videoFiles = allVideoFiles;
                const count = allVideoFiles.length;
                if (count > 0) {
                    const totalSize = allVideoFiles.reduce((sum, f) => sum + f.size, 0);
                    const sizeMB = (totalSize / (1024 * 1024)).toFixed(2);
                    document.getElementById('videoCount').textContent = 
                        count + ' video(s) selected (' + sizeMB + ' MB)';
                    document.getElementById('videoCount').style.color = '#4CAF50';
                    document.getElementById('videoCount').style.fontWeight = 'bold';
                }
            });
        }
        
        if (csvInput) {
            csvInput.addEventListener('change', function(e) {
                window._csvFile = e.target.files;
                if (e.target.files.length > 0) {
                    document.getElementById('csvIndicator').textContent = 
                        '✓ CSV file selected: ' + e.target.files[0].name;
                    document.getElementById('csvIndicator').style.color = '#4CAF50';
                    document.getElementById('csvIndicator').style.fontWeight = 'bold';
                }
            });
        }
    },
    on_finish: function(data) {
        // Store files for processing in next trial
        data.videoFiles = window._videoFiles;
        data.csvFiles = window._csvFile;
    }
};

const processFiles = {
    type: jsPsychCallFunction,
    async: true,
    func: function(done) {
        const videoFiles = window._videoFiles;
        const csvFiles = window._csvFile;
        
        if (!videoFiles || videoFiles.length === 0) {
            alert('Please select video files before continuing');
            return;
        }
        
        if (!csvFiles || csvFiles.length === 0) {
            alert('Please select a CSV file before continuing');
            return;
        }
        filesUploaded = true;
        // Create video blobs
        for (let i = 0; i < videoFiles.length; i++) {
            const file = videoFiles[i];
            const blobUrl = URL.createObjectURL(file);
            videoBlobs[file.name] = blobUrl;
        }
        
        // Process CSV
        uploadCSV(csvFiles[0])
            .then((csvResult) => {
                console.log(csvResult);
                csvResult.videos.forEach((video, idx) => {
                    if (videoBlobs.hasOwnProperty(video.videoFilename)) {
                        videosData.push(video)
                    }
                })
                
                // Find the last contiguous annotated video
                let lastAnnotatedIndex = -1;
                videosData.forEach((video, idx) => {
                    if (video.existingAnnotation) {
                        completedVideos.add(idx);
                        if (idx == lastAnnotatedIndex + 1) {
                            lastAnnotatedIndex = idx;
                        }
                    }
                });
                
                // Set current video to the one after the last annotated, or 0 if none annotated
                currentVideoIndex = lastAnnotatedIndex + 1;
                if (currentVideoIndex >= videosData.length) {
                    currentVideoIndex = 0;
                }
                
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
function getProgressBarHTML(videoIndex) {
    const completed = completedVideos.size;
    const total = videosData.length;
    const progress = (completed / total) * 100;
    
    return `
        <div style="background: #f5f5f5; padding: 15px; margin-bottom: 20px; border-radius: 8px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <strong>Video: ${videoIndex + 1} / ${total}</strong>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <label style="margin: 0; font-size: 14px;">Jump to video:</label>
                    <input type="number" id="jump-to-video" min="1" max="${total}" 
                           placeholder="${videoIndex + 1}" 
                           style="width: 80px; padding: 5px; border: 2px solid #2196F3; border-radius: 4px; font-size: 14px;">
                    <button id="jump-button" style="padding: 5px 15px; background: #2196F3; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 14px;">
                        Go
                    </button>
                    <button id="quit-button" style="padding: 5px 15px; background: #f44336; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 14px;">
                        Quit
                    </button>
                </div>
                <strong data-progress-completed>Completed: ${completed} / ${total}</strong>
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

function createVideoTrial(videoIndex) {
    const video = videosData[videoIndex];
    const videoUrl = videoBlobs[video.videoFilename];
    
    // Survey trial with video embedded
    const surveyTrial = {
            type: jsPsychSurvey,
            survey_json: {
              showQuestionNumbers: "off",
              elements: [
                {
                  type: "html",
                  name: "video_html",
                  html: `
                   ${getProgressBarHTML(videoIndex)}
                   <br>
                    <div style="max-width: 900px; margin: auto;">
                      <div style="background: #000; padding: 20px; border-radius: 8px; margin-bottom: 30px; text-align: center;">
                        <video width="800" controls autoplay style="max-width: 60%;">
                          <source src="${videoUrl}" type="video/mp4">
                          Your browser does not support the video tag.
                        </video>
                      </div>
                      <h4 style="text-align: center; margin-bottom: 30px;">Answer the following questions:</h4>
                    </div>`
                },
          
                /* ───────────────────────────────
                   1. What is the child doing?
                ─────────────────────────────── */
                {
                  type: "tagbox",
                  name: "childActivities",
                  title: "1. What is the child doing?",
                  placeholder: "Search activities...",
                  isRequired: true,
                  choices: dropdownOptions.childActivities,   
                },
                {
                  type: "dropdown",
                  name: "childActivityConfidence",
                  title: "1a. How confident are you?",
                  isRequired: true,
                  choices: [
                    { value: "1", text: "1 - Low" },
                    { value: "2", text: "2 - Medium" },
                    { value: "3", text: "3 - High" }
                  ]
                },
          
                /* ───────────────────────────────
                   2. Is another human present?
                ─────────────────────────────── */
                {
                  type: "dropdown",
                  name: "otherPersonPresent",
                  title: "2. Is there another human being?",
                  isRequired: true,
                  choices: ["yes", "no"]
                },
          
                {
                  type: "panel",
                  name: "other_person_panel",
                  visibleIf: "{otherPersonPresent} = 'yes'",
                  title: "Questions about the other person",
                  elements: [
                    {
                      type: "tagbox",
                      name: "otherPersonType",
                      title: "2a. Who is it?",
                      isRequired: true,
                      choices: dropdownOptions.otherPersonTypes
                    },
                    {
                      type: "tagbox",
                      name: "otherPersonActivities",
                      title: "2b. What are they doing?",
                      placeholder: "Search...",
                      choices: dropdownOptions.otherPersonActivities
                    },
                    {
                      type: "dropdown",
                      name: "otherPersonConfidence",
                      title: "2c. How confident are you?",
                      isRequired: true,
                      choices: [
                        { value: "1", text: "1 - Low" },
                        { value: "2", text: "2 - Medium" },
                        { value: "3", text: "3 - High" }
                      ]
                    },
                    {
                      type: "dropdown",
                      name: "sameSpace",
                      title: "2d. Are they in the same space as the child?",
                      choices: ["yes", "no"]
                    },
                    {
                      type: "dropdown",
                      name: "sameActivity",
                      title: "2e. Are they engaged in the same activity?",
                      choices: ["yes", "no"]
                    }
                  ]
                },
          
                /* ───────────────────────────────
                   3. Child posture
                ─────────────────────────────── */
                {
                  type: "tagbox",
                  name: "childPosture",
                  title: "3. Child posture?",
                  placeholder: "Search postures...",
                  isRequired: true,
                  choices: dropdownOptions.postures
                },
                {
                  type: "dropdown",
                  name: "childPostureConfidence",
                  title: "3a. Confidence?",
                  isRequired: true,
                  choices: [
                    { value: "1", text: "1 - Low" },
                    { value: "2", text: "2 - Medium" },
                    { value: "3", text: "3 - High" }
                  ]
                },
          
                /* ───────────────────────────────
                   4. Location
                ─────────────────────────────── */
                {
                  type: "tagbox",
                  name: "locations",
                  title: "4. Location?",
                  isRequired: true,
                  placeholder: "Search locations...",
                  choices: dropdownOptions.locations
                },
                {
                  type: "dropdown",
                  name: "locationConfidence",
                  title: "4a. Confidence?",
                  isRequired: true,
                  choices: [
                    { value: "1", text: "1 - Low" },
                    { value: "2", text: "2 - Medium" },
                    { value: "3", text: "3 - High" }
                  ]
                }
              ]
            },
            choices: ["Save & Continue", "Skip"],
            survey_function: function(survey) {
                survey.data = video.existingAnnotation
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
                        
                        const targetIndex = targetVideoNum - 1; // Convert to 0-based index
                        
                        if (targetIndex !== videoIndex) {
                            // End current timeline and rebuild from target
                            jsPsych.endCurrentTimeline();
                            new_timeline = []
                            // Add videos starting from target index
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
                            // End current timeline and jump to final screen
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
                }, 100);
            },
        on_finish: async function(data) {            
            // Get the data that was stored during on_load
            const trialData = data.response;
            
            // Build annotation data object
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
                // Update progress bar dynamically
                updateProgressBar();
                console.log('Saved:', video.videoFilename);
            } catch (error) {
                alert('Failed to save: ' + error.message);
            }
        }
    };
    
    const surveyDescriptionTrial = {
        type: jsPsychSurvey,
        survey_json: {
          showQuestionNumbers: "off",
          elements: [
            {
              type: "html",
              name: "video_html",
              html: `
               ${getProgressBarHTML(videoIndex)}
               <br>
                <div style="max-width: 900px; margin: auto;">
                  <div style="background: #000; padding: 20px; border-radius: 8px; margin-bottom: 30px; text-align: center;">
                    <video width="800" controls autoplay style="max-width: 60%;">
                      <source src="${videoUrl}" type="video/mp4">
                      Your browser does not support the video tag.
                    </video>
                  </div>
      
                  <div style="background: #e8f5e9; padding: 20px; border-radius: 8px; border-left: 5px solid #4CAF50; margin-bottom: 30px;">
                    <h3 style="margin-top: 0;">Video Description</h3>
                    <p style="font-size: 16px; line-height: 1.6;">
                      ${video.description || "No description available"}
                    </p>
                  </div>
      
                  <h4 style="text-align: center; margin-bottom: 30px;">Answer the following questions:</h4>
                </div>`
            },
      
            /* ───────────────────────────────
               5. Rate description
            ─────────────────────────────── */
            {
              type: "rating",
              name: "descriptionRating",
              title: "Rate description (1–5)",
              isRequired: true,
              minRateDescription: "Poor",
              maxRateDescription: "Excellent",
              rateMin: 1,
              rateMax: 5
            }
          ]
        },
        survey_function: function(survey) {
            console.log(video.existingAnnotation)
            survey.data = video.existingAnnotation
        },
        choices: ["Save & Continue", "Skip"],
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
                    
                    const targetIndex = targetVideoNum - 1; // Convert to 0-based index
                    
                    if (targetIndex !== videoIndex) {
                        // End current timeline and rebuild from target
                        jsPsych.endCurrentTimeline();
                        new_timeline = []
                        // Add videos starting from target index
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
                        // End current timeline and jump to final screen
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
            }, 100);
        },
        on_finish: async function(data) {
            // Get the data that was stored during on_load
            const trialData = data.response;
            
            // Build annotation data object
            const annotationData = {
                videoFilename: video.videoFilename,
                description: video.description,
                ...trialData
            };
            
            try {
                await saveAnnotation(annotationData);
                completedVideos.add(videoIndex);
                showSaveIndicator();
                video.existingAnnotation = annotationData;
                // Update progress bar dynamically
                updateProgressBar();
                console.log('Saved:', video.videoFilename);
            } catch (error) {
                alert('Failed to save: ' + error.message);
            }
        }
    };

    return {
        timeline: [surveyTrial, surveyDescriptionTrial]
    };
}

// Run the experiment
jsPsych.run(timeline);