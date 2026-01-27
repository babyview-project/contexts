// activities.js - jsPsych Activity Annotation Experiment

// Configuration
const CONFIG = {
    API_BASE: 'http://stanford-cogsci.org:3000/api',
    MAX_TRAINING_ATTEMPTS: 3,
    TRAINING_VIDEOS_PER_ATTEMPT: 5,
    TRAINING: true,
    PERCENTAGE_TO_PASS: 0.75,
    AUTH: true
};

// Get URL parameter for task type
const urlParams = new URLSearchParams(window.location.search);
const taskType = urlParams.get('type') || 'do'; // default to 'do' if not specified
const isSeeing = taskType === 'see';

console.log('Task type:', taskType, 'isSeeing:', isSeeing);

// Global state
let credentials = { username: '', password: '' };
let annotatorName = '';
let videosData = [];
let videoBlobs = {}; // Store blob URLs for videos in browser memory
let currentVideoIndex = 0;
let dropdownOptions = {};
let completedVideos = new Set();
let trainingAttempts = 0;
let currentTrainingVideos = [];
let allGoldStandardVideos = []; // Store all training videos
let usedTrainingVideoIndices = new Set();
let currentTrainingAttemptStartIndex = 0;

// Session management - FIXED
const AUTH_KEY = 'activity_annotations_authenticated';

function isAuthenticated() {
    return sessionStorage.getItem(AUTH_KEY) === 'true';
}

function setAuthenticated(value) {
    sessionStorage.setItem(AUTH_KEY, value ? 'true' : 'false');
}

function clearAuthentication() {
    sessionStorage.removeItem(AUTH_KEY);
}

// API call helper
async function apiCall(endpoint, options = {}) {
    const defaultOptions = {
        credentials: 'include', // Important for session cookies
        headers: {
            'Content-Type': 'application/json'
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
                clearAuthentication();
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

// API functions
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

async function getExampleVideo() {
    // Pass taskType to get appropriate examples
    return apiCall(`/training/example-video?taskType=${taskType}`);
}

async function getGoldStandardVideos() {
    // Pass taskType to get appropriate gold standards
    return apiCall(`/training/gold-standard-videos?taskType=${taskType}`);
}

async function exportAnnotations() {
    try {
        const response = await fetch(CONFIG.API_BASE + '/export', {
            method: 'GET',
            credentials: 'include'
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
        completedText.style.marginLeft = '15px';
    }
}

// Initialize jsPsych
const jsPsych = initJsPsych({
    display_element: 'jspsych-target',
    on_finish: function() {
        jsPsych.data.displayData();
    }
});

// ============================================================================
// AUTHENTICATION - FIXED
// ============================================================================

let loginSuccess = false;
let needsLogin = true;

const authTrial = {
    type: jsPsychSurveyHtmlForm,
    preamble: '<h2 style="text-align: center; margin-bottom: 30px;">Login to activity annotations</h2>',
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
        fetch(CONFIG.API_BASE + '/auth/login', {
            method: 'POST',
            credentials: 'include',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                username: credentials.username,
                password: credentials.password
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Authentication failed');
            }
            return response.json();
        })
        .then(() => {
            loginSuccess = true;
            setAuthenticated(true);
            console.log('✓ Authentication successful');
            credentials.username = '';
            credentials.password = '';
            done({ success: true });
        })
        .catch((err) => {
            clearAuthentication();
            loginSuccess = false;
            credentials.username = '';
            credentials.password = '';
            
            console.error("Authentication failed:", err.message);
            
            const errorMsg = document.createElement('div');
            errorMsg.style.cssText = 'color: red; text-align: center; margin-top: 20px; font-size: 16px;';
            errorMsg.textContent = 'Authentication failed. Please try again.';
            document.querySelector('#jspsych-target').prepend(errorMsg);
            
            setTimeout(() => {
                if (errorMsg.parentNode) {
                    errorMsg.remove();
                }
            }, 3000);
            
            done({ success: false });
        });
    }
};

const verifySession = {
    type: jsPsychCallFunction,  
    async: true,               
    func: function(done) {     
        if (!CONFIG.AUTH) {
            needsLogin = false;
            loginSuccess = true;
            done();            
            return;
        }
        
        console.log('Verifying existing session...');
        apiCall('/health')
            .then(() => {
                loginSuccess = true;
                needsLogin = false;
                setAuthenticated(true);
                console.log('✓ Existing session verified');
                done();         
            })
            .catch((err) => {
                console.log('✗ Session invalid or expired, prompting for login');
                clearAuthentication();
                loginSuccess = false;
                needsLogin = true;
                done();         
            });
    }
};

const loginLoop = {
    timeline: [authTrial, checkAuth],
    loop_function: function() {
        return !loginSuccess;
    }
};

// ============================================================================
// BUILD TIMELINE
// ============================================================================

let timeline = [];

// Authentication flow
if (CONFIG.AUTH) {
    timeline.push(verifySession);
    timeline.push({
        timeline: [loginLoop],
        conditional_function: function() {
            return needsLogin;
        }
    });
}

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
        }
    }
};

timeline.push(nameScreen);

const consentScreen = {
    type: jsPsychHtmlButtonResponse,
    stimulus: `
        <div style="max-width: 800px; margin: auto; padding: 40px; background: white; border-radius: 12px;">
            <h2 style="text-align: center; margin-bottom: 30px;">Consent Agreement</h2>
            <div style="font-size: 16px; line-height: 1.8; text-align: left; padding: 20px; background: #f5f5f5; border-radius: 8px;">
                <p><strong>By participating in this task, you agree that you will:</strong></p>
                <ul style="margin: 20px 0;">
                    <li>NOT attempt to reidentify individuals who appear in the stimuli</li>
                    <li>NOT take screenshots of the stimuli</li>
                    <li>NOT redistribute the stimuli in any form</li>
                </ul>
                <p style="margin-top: 20px;">
                    By clicking "I Agree" below, you confirm that you understand and agree to these terms.
                </p>
            </div>
        </div>
    `,
    choices: ['I Agree', 'I Do Not Agree'],
    on_finish: function(data) {
        if (data.response === 1) {
            // User did not consent
            alert('You must agree to the terms to participate in this study.');
            jsPsych.endExperiment('User did not consent to terms');
        }
    }
};

timeline.push(consentScreen);

const skipTrainingScreen = {
    type: jsPsychHtmlButtonResponse,
    stimulus: `
        <div style="max-width: 600px; margin: auto; padding: 40px; background: white; border-radius: 12px; text-align: center;">
            <h3>Training Phase</h3>
            <p style="font-size: 18px; margin: 20px 0;">
                You're about to begin the training phase. This will help you learn how to annotate videos correctly.
            </p>
            <p style="font-size: 16px; color: #666;">
                If you've already completed training/want to go directly to the main task, click "Skip Training".
            </p>
        </div>
    `,
    choices: ['Start Training', 'Skip Training'],
    on_finish: function(data) {
        if (data.response === 1) {
            // User chose to skip - set flag
            window.skipTraining = true;
        } else {
            window.skipTraining = false;
        }
    }
};

timeline.push(skipTrainingScreen);

// ============================================================================
// TRAINING PHASE
// ============================================================================

// Introduction to training
const trainingIntro = {
    type: jsPsychHtmlButtonResponse,
    stimulus: `
        <div style="max-width: 800px; margin: auto; padding: 40px; background: white; border-radius: 12px;">
            <h2>Training Phase</h2>
            <p style="font-size: 18px; line-height: 1.6;">
                Before you begin annotating videos, you'll go through a brief training phase:
            </p>
            <ol style="font-size: 16px; line-height: 1.8; text-align: left;">
                <li><strong>Example Videos:</strong> First, you'll watch example videos with the correct annotations shown for each activity.</li>
                <li><strong>Practice Videos:</strong> Then, you'll annotate ${CONFIG.TRAINING_VIDEOS_PER_ATTEMPT} practice videos to test your understanding.</li>
                <li><strong>Feedback:</strong> You must score at least ${CONFIG.PERCENTAGE_TO_PASS * 100}% to proceed. If not, we'll show you the correct answers and provide 5 new videos.</li>
                <li><strong>Maximum Attempts:</strong> You have up to 3 attempts to pass the training.</li>
            </ol>
            <p style="font-size: 16px; margin-top: 20px;">
                Click below to begin with an example video.
            </p>
        </div>
    `,
    choices: ['Start Training']
};

// Example video walkthrough
const exampleVideoTrial = {
    type: jsPsychCallFunction,
    async: true,
    func: async function(done) {
        try {
            const exampleData = await getExampleVideo();
            
            if (!exampleData.success || !exampleData.videos || exampleData.videos.length === 0) {
                alert('Failed to load example videos');
                done({ success: false });
                return;
            }
            
            // Add a screen for EACH example video
            exampleData.videos.forEach((video, index) => {
                const videoUrl = `${CONFIG.API_BASE.replace('/api', '')}/training-videos/${video.videoFilename}`;
                
                const exampleScreen = {
                    type: jsPsychHtmlButtonResponse,
                    stimulus: `
                        <style>
                            .reasoning { font-size: 14px; color: #555; }
                        </style>
                        <div style="max-width: 1000px; margin: auto; padding: 10px;">
                            <h3>Example of ${video.primaryActivity}</h3>
                            <p style="font-size: 16px; margin-bottom: 20px;">
                                Watch this example and review the correct annotations and reasoning below.<br>
                                Example video ${index+1} of ${exampleData.videos.length} 
                            </p>
                            
                            <div style="display: flex; gap: 20px;">
                                <div style="flex: 1;">
                                    <video controls autoplay style="width: 100%; max-height: 60vh; background: #000;">
                                        <source src="${videoUrl}" type="video/mp4">
                                    </video>
                                </div>
                                
                                <div style="flex: 1; background: #f5f5f5; padding: 20px; border-radius: 8px; max-height: 60vh; overflow-y: auto;">
                                    <h3>Correct Annotations:</h3>
                                    <div style="text-align: left; line-height: 2; font-size: 16px;">
                                        <p><strong>${isSeeing ? 'Primary thing the child is seeing' : 'Primary activity that the child is doing'}:</strong> ${video.primaryActivity}</p>
                                        <p class="reasoning">Reasoning: ${video.primaryActivityReasoning}</p>
                                        <p><strong>Confidence:</strong> ${video.primaryActivityConfidence}/3</p>
                                        <p><strong>${isSeeing ? 'Other things the child is seeing' : 'Other activities that the child is doing'}:</strong> ${video.otherActivities.join(', ') || 'none'}</p>
                                        <p class="reasoning">Reasoning: ${video.otherActivitiesReasoning}</p>
                                        <p><strong>Other activities confidence:</strong> ${video.otherActivitiesConfidence}/3</p>
                                        <p><strong>Anyone interacting:</strong> ${video.anyoneInteracting}</p>
                                        <p class="reasoning">Reasoning: ${video.interactingReasoning}</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    `,
                    choices: index === exampleData.videos.length - 1 
                        ? ['Continue to Practice Videos'] 
                        : ['Next Example']
                };
                
                jsPsych.addNodeToEndOfTimeline(exampleScreen);
            });
            
            done({ success: true });
            
        } catch (error) {
            alert('Error loading example videos: ' + error.message);
            done({ success: false });
        }
    }
};

// Store training responses globally
let trainingResponses = [];

// Load gold standard videos for training
const loadTrainingVideos = {
    type: jsPsychCallFunction,
    async: true,
    func: async function(done) {
        try {
            const goldData = await getGoldStandardVideos();
            console.log(goldData)
            if (!goldData.success || !goldData.videos || goldData.videos.length < CONFIG.TRAINING_VIDEOS_PER_ATTEMPT) {
                alert('Not enough gold standard videos available for training');
                done({ success: false });
                return;
            }
            
            // Randomly select videos for this attempt
            const shuffled = goldData.videos.sort(() => Math.random() - 0.5);
            currentTrainingVideos = shuffled.slice(0, CONFIG.TRAINING_VIDEOS_PER_ATTEMPT);
            
            // Reset training responses for this attempt
            trainingResponses = [];
            
            console.log('Loaded training videos:', currentTrainingVideos.length);
            done({ success: true });
            
        } catch (error) {
            alert('Failed to load training videos: ' + error.message);
            done({ success: false });
        }
    }
};

const loadAllTrainingVideos = {
    type: jsPsychCallFunction,
    async: true,
    func: async function(done) {
        try {
            const goldData = await getGoldStandardVideos();
            console.log('Gold data received:', goldData);
            
            if (!goldData.success || !goldData.videos || goldData.videos.length < 15) {
                alert('Not enough gold standard videos available for training (need 15)');
                done({ success: false });
                return;
            }

            goldData.videos = goldData.videos.sort(() => Math.random() - 0.5);
            
            // Store ALL 15 videos
            allGoldStandardVideos = goldData.videos.slice(0, 15);
            console.log(`Loaded ${allGoldStandardVideos.length} gold standard videos`);
            
            // Ensure options are loaded
            if (!dropdownOptions.activities || dropdownOptions.activities.length === 0) {
                console.log('Loading dropdown options...');
                dropdownOptions = await getOptions();
            }
            
            done({ success: true });
            
        } catch (error) {
            alert('Failed to load training videos: ' + error.message);
            done({ success: false });
        }
    }
};

// Create the ENTIRE training timeline upfront (all 15 videos + 3 feedback screens)
const createCompleteTrainingTimeline = {
    type: jsPsychCallFunction,
    async: true,
    func: async function(done) {
        try {
            console.log('Creating complete training timeline with all 15 videos...');
            
            // Create 3 attempts, each with 5 videos + feedback
            for (let attempt = 0; attempt < CONFIG.MAX_TRAINING_ATTEMPTS; attempt++) {
                const startIdx = attempt * CONFIG.TRAINING_VIDEOS_PER_ATTEMPT;
                const endIdx = startIdx + CONFIG.TRAINING_VIDEOS_PER_ATTEMPT;
                
                console.log(`Creating attempt ${attempt + 1}: videos ${startIdx} to ${endIdx - 1}`);
                
                // Add 5 video trials for this attempt
                for (let i = startIdx; i < endIdx; i++) {
                    const videoData = allGoldStandardVideos[i];
                    const videoTrial = createVideoTrial(i - startIdx, true, videoData);
                    
                    // Add conditional: only show if we're on this attempt
                    const conditionalTrial = {
                        timeline: [videoTrial],
                        conditional_function: function() {
                            const shouldShow = currentTrainingAttemptStartIndex === startIdx;
                            console.log(`Video ${i}: attempt ${attempt + 1}, shouldShow=${shouldShow} (currentStart=${currentTrainingAttemptStartIndex})`);
                            return shouldShow;
                        }
                    };
                    
                    jsPsych.addNodeToEndOfTimeline(conditionalTrial);
                }
                
                // Add feedback screen for this attempt
                const feedbackTrial = {
                    timeline: [trainingFeedback],
                    conditional_function: function() {
                        const shouldShow = currentTrainingAttemptStartIndex === startIdx;
                        console.log(`Feedback for attempt ${attempt + 1}: shouldShow=${shouldShow}`);
                        return shouldShow;
                    }
                };
                
                jsPsych.addNodeToEndOfTimeline(feedbackTrial);
            }
            
            console.log('Complete training timeline created');
            done({ success: true });
            
        } catch (error) {
            console.error('Error creating training timeline:', error);
            alert('Failed to create training timeline: ' + error.message);
            done({ success: false });
        }
    }
};

// Updated training feedback that manages attempt progression
const trainingFeedback = {
    type: jsPsychHtmlButtonResponse,
    stimulus: function() {
        // Calculate which attempt we're on
        const attemptNumber = (currentTrainingAttemptStartIndex / CONFIG.TRAINING_VIDEOS_PER_ATTEMPT) + 1;
        
        // Wait for all responses before checking
        if (trainingResponses.length < CONFIG.TRAINING_VIDEOS_PER_ATTEMPT) {
            console.warn('Not all training responses collected yet', {
                have: trainingResponses.length,
                need: CONFIG.TRAINING_VIDEOS_PER_ATTEMPT
            });
            return `<div style="text-align: center; padding: 40px;">
                <p>Loading results...</p>
            </div>`;
        }
        
        const results = checkTrainingResults();
        let feedbackHTML = '';
        if (results.pcCorrect >= CONFIG.PERCENTAGE_TO_PASS) {
            feedbackHTML = `
                <div style="max-width: 800px; margin: auto; padding: 40px; background: white; border-radius: 12px; text-align: center;">
                    <h2 style="color: #4CAF50;">✅ Training Passed!</h2>
                    <p style="font-size: 18px;">
                        Congratulations! You scored ${Math.round(results.pcCorrect*100*100)/100}%.
                    </p>
                    <p style="font-size: 16px;">
                        You're ready to begin the main annotation task. Please review the feedback below.
                    </p>
                </div>
            `;
        } else {
            feedbackHTML = `
                <div style="max-width: 1200px; margin: auto; padding: 40px; background: white; border-radius: 12px;">
                    <h2 style="color: #f44336;">Training Not Passed</h2>
                    <p style="font-size: 16px;">
                        Attempt ${attemptNumber} of ${CONFIG.MAX_TRAINING_ATTEMPTS} complete.
                        You got ${results.results.filter(r => r.correct).length} out of ${results.results.length} correct.
                        Please review the correct answers below.
                    </p>
            `;
        }
            
            results.results.forEach((result, idx) => {
                const status = result.correct ? '✅' : '❌';
                const backgroundColor = result.correct ? '#e8f5e9' : '#ffebee';
                const videoUrl = `${CONFIG.API_BASE.replace('/api', '')}/training-videos/${result.correctAnswers.videoFilename}`;
                
                feedbackHTML += `
                    <div style="margin: 20px 0; padding: 15px; background: ${backgroundColor}; border-radius: 8px;">
                        <h3>${status} Video ${idx + 1}</h3>
                        <div style="display: grid; grid-template-columns: 200px 1fr 1fr; gap: 20px; align-items: start;">
                            <div style="text-align: center;">
                                <video controls style="width: 100%; max-height: 150px; background: #000; border-radius: 4px;">
                                    <source src="${videoUrl}" type="video/mp4">
                                </video>
                            </div>
                            <div>
                                <h4>Your Answers:</h4>
                                <p><strong>Primary:</strong> ${result.userAnswers.primaryActivity || 'N/A'}</p>
                                <p><strong>Other:</strong> ${(result.userAnswers.otherActivities || []).filter(a => a !== 'none').join(', ') || 'none'}</p>
                                <p><strong>Interacting:</strong> ${result.userAnswers.anyoneInteracting || 'N/A'}</p>
                            </div>
                            <div>
                                <h4>Correct Answers:</h4>
                                <p><strong>Primary:</strong> ${result.correctAnswers.primaryActivity}</p>
                                <p><strong>Other:</strong> ${(result.correctAnswers.otherActivities || []).join(', ') || 'none'}</p>
                                <p><strong>Interacting:</strong> ${result.correctAnswers.anyoneInteracting}</p>
                            </div>
                        </div>
                        <div style="margin-top: 10px; font-size: 12px; color: #666;">
                            ${!result.details.primaryCorrect ? '❌ Primary activity incorrect. ' : ''}
                            ${!result.details.interactingCorrect ? '❌ Interaction incorrect. ' : ''}
                        </div>
                    </div>
                `;
            });
            
            if (results.pcCorrect < CONFIG.PERCENTAGE_TO_PASS) {
            if (attemptNumber < CONFIG.MAX_TRAINING_ATTEMPTS) {
                feedbackHTML += `
                    <p style="font-size: 16px; margin-top: 20px;">
                        You will now see ${CONFIG.TRAINING_VIDEOS_PER_ATTEMPT} new practice videos.
                    </p>
                `;
            } else {
                feedbackHTML += `
                    <p style="font-size: 16px; margin-top: 20px; color: #f44336;">
                        You have used all ${CONFIG.MAX_TRAINING_ATTEMPTS} attempts. 
                        The experiment will now end. Please contact the researcher.
                    </p>
                `;
            }
            }
            
            feedbackHTML += `</div>`;
            
            return feedbackHTML;
    },
    choices: function() {
        const attemptNumber = (currentTrainingAttemptStartIndex / CONFIG.TRAINING_VIDEOS_PER_ATTEMPT) + 1;
        
        // Safety check
        if (trainingResponses.length < CONFIG.TRAINING_VIDEOS_PER_ATTEMPT) {
            return ['Continue'];
        }
        
        const results = checkTrainingResults();
        if (results.pcCorrect >= CONFIG.PERCENTAGE_TO_PASS) {
            return ['Begin Main Task'];
        } else if (attemptNumber < CONFIG.MAX_TRAINING_ATTEMPTS) {
            return ['Try Again'];
        } else {
            return ['End Experiment'];
        }
    },
    on_finish: function() {
        const attemptNumber = (currentTrainingAttemptStartIndex / CONFIG.TRAINING_VIDEOS_PER_ATTEMPT) + 1;
        
        // Safety check
        if (trainingResponses.length < CONFIG.TRAINING_VIDEOS_PER_ATTEMPT) {
            console.error('Feedback shown before all responses collected!');
            return;
        }
        
        const results = checkTrainingResults();
        
        console.log('Training attempt complete:', {
            attempt: attemptNumber,
            allCorrect: results.allCorrect,
            correctCount: results.results.filter(r => r.correct).length,
            totalCount: results.results.length
        });
        
        if (results.pcCorrect >= CONFIG.PERCENTAGE_TO_PASS) {
            console.log('Training passed! Moving to main task.');
            // Training passed - continue to main task
        } else if (attemptNumber >= CONFIG.MAX_TRAINING_ATTEMPTS) {
            console.log('Training failed after max attempts. Ending experiment.');
            jsPsych.endExperiment('Training failed after maximum attempts');
        } else {
            // Move to next set of 5 videos
            currentTrainingAttemptStartIndex += CONFIG.TRAINING_VIDEOS_PER_ATTEMPT;
            trainingResponses = []; // Reset for next attempt
            console.log(`Moving to attempt ${attemptNumber + 1}, starting at index ${currentTrainingAttemptStartIndex}`);
        }
    }
};

// Updated checkTrainingResults to work with current set
function checkTrainingResults() {
    // Safety check
    if (!trainingResponses || trainingResponses.length === 0) {
        console.warn('No training responses available yet');
        return { allCorrect: false, results: [] };
    }
    
    if (trainingResponses.length !== CONFIG.TRAINING_VIDEOS_PER_ATTEMPT) {
        console.warn('Mismatch between responses and expected count', {
            responses: trainingResponses.length,
            expected: CONFIG.TRAINING_VIDEOS_PER_ATTEMPT
        });
    }
    
    let numCorrect = 0;
    const results = [];
    
    // Get the correct videos for this attempt
    const startIdx = currentTrainingAttemptStartIndex;
    const endIdx = startIdx + CONFIG.TRAINING_VIDEOS_PER_ATTEMPT;
    const currentAttemptVideos = allGoldStandardVideos.slice(startIdx, endIdx);
    
    for (let i = 0; i < CONFIG.TRAINING_VIDEOS_PER_ATTEMPT; i++) {
        const correct = currentAttemptVideos[i];
        const response = trainingResponses[i];
        
        // Skip if response doesn't exist yet
        if (!response) {
            console.warn(`No response for training video ${i}`);
            allCorrect = false;
            continue;
        }
        
        const primaryCorrect = response.primaryActivity === correct.primaryActivity;
        
        // Handle other activities - normalize to arrays and sort
        console.log(response.otherActivities);
        console.log(correct.otherActivities);
        const responseOtherActivities = Array.isArray(response.otherActivities) 
            ? response.otherActivities.filter(a => a !== 'none').sort()
            : [];
        const correctOtherActivities = Array.isArray(correct.otherActivities)
            ? correct.otherActivities.sort()
            : [];
        
        const otherActivitiesCorrect = JSON.stringify(responseOtherActivities) === 
                                      JSON.stringify(correctOtherActivities);
        
        const interactingCorrect = response.anyoneInteracting === correct.anyoneInteracting;
        
        const videoCorrect = primaryCorrect && interactingCorrect;
        
        results.push({
            videoIndex: i,
            correct: videoCorrect,
            details: {
                primaryCorrect,
                otherActivitiesCorrect,
                interactingCorrect
            },
            correctAnswers: correct,
            userAnswers: response
        });
        
        if (videoCorrect) {
            numCorrect++;
        }
    }

    let pcCorrect = numCorrect/CONFIG.TRAINING_VIDEOS_PER_ATTEMPT
    return { pcCorrect, results };
}

function createTrainingTimeline() {
    console.log('Creating training timeline for', currentTrainingVideos.length, 'videos');
    
    const trainingTrials = [];
    for (let i = 0; i < currentTrainingVideos.length; i++) {
        const trial = createVideoTrial(i, true, currentTrainingVideos[i]);
        trainingTrials.push(trial);
    }
    
    return {
        timeline: trainingTrials
    };
}

timeline.push({
    timeline: [trainingIntro, loadAllTrainingVideos, exampleVideoTrial, createCompleteTrainingTimeline],
    conditional_function: function() {
        return !window.skipTraining;
    }
});

// ============================================================================
// MAIN ANNOTATION TASK
// ============================================================================

// Global file storage (must be outside trial definition)
window._videoFiles = null;
let filesUploaded = false;

const loadVideos = {
    type: jsPsychCallFunction,
    async: true,
    func: function(done) {
        // Fetch video list from server
        apiCall('/video-list')
            .then(async (result) => {
                const videoFilenames = result.videos;
                
                if (videoFilenames.length === 0) {
                    alert('No videos found in sampled_context_videos directory');
                    done({ success: false });
                    return;
                }
                
                // Build video data (videos will be streamed from server)
                videoFilenames.forEach((filename, idx) => {
                    videosData.push({
                        videoFilename: filename,
                        videoUrl: `${CONFIG.API_BASE.replace('/api', '')}/videos/${filename}`,
                        description: '',
                        order: idx + 1
                    });
                });
                
                // Get existing annotations
                const annotationData = await getAnnotationsForVideos(videoFilenames);
                const annotationMap = annotationData.annotations;
                
                // Mark completed videos
                videosData.forEach((video, idx) => {
                    if (annotationMap[video.videoFilename]) {
                        video.existingAnnotation = annotationMap[video.videoFilename];
                        completedVideos.add(idx);
                    }
                });
                
                // Find starting point
                let lastAnnotatedIndex = -1;
                for (let i = 0; i < videosData.length; i++) {
                    if (completedVideos.has(i)) {
                        lastAnnotatedIndex = i;
                    } else {
                        break;
                    }
                }
                
                currentVideoIndex = lastAnnotatedIndex + 1;
                if (currentVideoIndex >= videosData.length) {
                    currentVideoIndex = 0;
                }
                
                console.log(`Loaded ${videosData.length} videos from server, starting at index ${currentVideoIndex}`);
                done({ success: true });
            })
            .catch((error) => {
                alert('Failed to load videos: ' + error.message);
                done({ success: false });
            });
    }
};

timeline.push(loadVideos);

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
    on_load: async function() {
        // Ensure options are loaded before building timeline
        if (!dropdownOptions.activities) {
            console.log('Options not loaded yet, fetching...');
            try {
                dropdownOptions = await getOptions();
            } catch (error) {
                console.error('Failed to load options:', error);
                alert('Failed to load dropdown options');
                return;
            }
        }
        
        console.log('Activities loaded:', dropdownOptions.activities);
        console.log(videosData.length)
        
        // Add videos starting from currentVideoIndex
        for (let i = currentVideoIndex; i < videosData.length; i++) {
            const videoTrialSet = createVideoTrial(i, false);
            jsPsych.addNodeToEndOfTimeline(videoTrialSet);
        }
        
        // Add remaining videos from the beginning if we didn't start at 0
        if (currentVideoIndex > 0) {
            for (let i = 0; i < currentVideoIndex; i++) {
                const videoTrialSet = createVideoTrial(i, false);
                jsPsych.addNodeToEndOfTimeline(videoTrialSet);
            }
        }
        
        jsPsych.addNodeToEndOfTimeline(finalScreen);
    }
};
timeline.push(buildVideoTimeline);

// Create video annotation trials
function createVideoTrial(videoIndex, isTraining = false, trainingVideoData = null) {
    const video = isTraining ? trainingVideoData : videosData[videoIndex];
    const videoUrl = isTraining 
        ? `${CONFIG.API_BASE.replace('/api', '')}/training-videos/${video.videoFilename}`
        : video.videoUrl;
    
    const totalVideos = isTraining ? CONFIG.TRAINING_VIDEOS_PER_ATTEMPT : videosData.length;
    const currentNum = isTraining ? (videoIndex + 1) : (videoIndex + 1);
    
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
                                height: 62vh;
                            }
                            .video-section {
                                flex: 0 0 60%;
                                display: flex;
                                flex-direction: column;
                                gap: 5px;
                            }
                            .questions-section {
                                flex: 0 0 40%;
                                padding: 10px;
                                background: #f5f5f5;
                                border-radius: 8px;
                                overflow-y: auto;
                                max-height: 60vh;
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
                                max-height: 60vh;
                                aspect-ratio: 512 / 910;
                            }
                            .video-player video {
                                width: 100%;
                                max-height: 58vh;
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
                            .sd-action-bar {
                                padding-top: 8px !important;
                                margin-top: 8px !important;
                            }
                            .sd-btn {
                                padding: 8px 16px !important;
                                font-size: 14px !important;
                            }
                            html, body {
                                overflow: hidden;
                                padding-top: 20px;
                            }   
                            element.style {
                                padding-top: 20px !important;
                        }   
                            .sd-error,
                            .sd-element__erbox,
                            .sd-question__erbox,
                            .sd-element__erbox--above-element,
                            .sd-question__erbox--above-question {
                                position: relative !important;
                                float: none !important;
                                clear: both !important;
                                width: 100% !important;
                                margin: 4px 0 !important;
                                padding: 4px 8px !important;
                                background: #ffebee !important;
                                color: #c62828 !important;
                                border-radius: 4px !important;
                                font-size: 12px !important;
                                line-height: 1.4 !important;
                            }
                        </style>
                        
                        <div class="video-annotation-container">
                            <div class="video-section">
                                <div class="progress-bar-container">
                                    ${isTraining ? `
                                        <div style="text-align: center;">
                                            <strong style="font-size:15px;">Training Video ${currentNum} of ${totalVideos}</strong>
                                        </div>
                                    ` : `
                                        <div class="progress-controls">
                                            <strong>Video: ${currentNum} / ${totalVideos}</strong>
                                            <div class="jump-controls">
                                                <label style="margin: 0; font-size: 14px;">Jump to:</label>
                                                <input type="number" id="jump-to-video" min="1" max="${totalVideos}" 
                                                       placeholder="${currentNum}">
                                                <button class="jump-button" id="jump-button">Go</button>
                                                <button class="quit-button" id="quit-button">Quit</button>
                                                <br/>
                                                <p></p>
                                            </div>
                                            <br/>
                                            <div>
                                            <strong data-progress-completed style="font-size:15px;">Completed: ${completedVideos.size} / ${totalVideos}</strong>
                                            </div>
                                        </div>
                                        <div class="progress-bar">
                                            <div class="progress-bar-fill" id="progress-bar-fill" style="width: ${(completedVideos.size / totalVideos) * 100}%"></div>
                                            <div class="progress-text">${Math.round((completedVideos.size / totalVideos) * 100)}%</div>
                                        </div>
                                    `}
                                </div>
                                
                                <div class="video-player">
                                    <video controls autoplay>
                                        <source src="${videoUrl}" type="video/mp4">
                                        Your browser does not support the video tag.
                                    </video>
                                </div>
                            </div>
                            
                            <div class="questions-section" id="questions-container-${isTraining ? 'training-' : ''}${videoIndex}">
                                <h3 style="margin-top: 0;">Annotation Questions</h3>
                            </div>
                        </div>
                        
                        <div id="questions-placeholder"></div>
                    `
                },
                
                // Questions (same for both training and main)
                {
                    type: "dropdown",
                    name: "primaryActivity",
                    title: isSeeing 
                        ? "1. What is the primary thing that the child wearing the camera is seeing?"
                        : "1. What is the primary activity that the child wearing the camera is doing?",
                    placeholder: isSeeing ? "Select primary thing seen..." : "Select primary activity...",
                    isRequired: true,
                    choices: dropdownOptions.activities,   
                },
                {
                    type: "text",
                    name: "primaryActivityOther",
                    title: isSeeing 
                        ? "Please specify the primary thing seen:"
                        : "Please specify the primary activity:",
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
                    title: isSeeing
                        ? "3. What other things is the child wearing the camera seeing? [multi-select]"
                        : "3. What other activities is the child wearing the camera doing? [multi-select]",
                    placeholder: "Search and select...",
                    isRequired: true,
                    choices: [...dropdownOptions.activities, "none"]
                },
                {
                    type: "text",
                    name: "otherActivitiesOther",
                    title: isSeeing
                        ? "Please specify the other things seen (comma-separated if multiple):"
                        : "Please specify the other activities (comma-separated if multiple):",
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
        data: {
            isTraining: isTraining,
            videoIndex: videoIndex,
            ...(isTraining && { correctAnswers: trainingVideoData })
        },
        survey_function: function(survey) {
            // Load existing data for main task
            if (!isTraining && video.existingAnnotation) {
                survey.data = video.existingAnnotation;
            }
            
            // Handle conditional visibility
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
                    const containerID = isTraining ? `questions-container-training-${videoIndex}` : `questions-container-${videoIndex}`;
                    const questionsSection = document.getElementById(containerID);
                    const questionsPlaceholder = document.getElementById('questions-placeholder');
                    
                    if (!questionsSection) {
                        console.error('Could not find questions section:', containerID);
                        return;
                    }
                    
                    // Get all question rows (skip the first HTML element)
                    const allRows = document.querySelectorAll('.sd-row');
                    console.log('Found rows:', allRows.length, 'for', isTraining ? 'training' : 'main', 'video', videoIndex);
                    
                    // Move questions (all rows except first one which is video/progress)
                    allRows.forEach((row, index) => {
                        if (index > 0) { // Skip first row (video HTML)
                            if (row.parentNode && row.parentNode !== questionsSection) {
                                questionsSection.appendChild(row);
                            }
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
                    const containerID = isTraining ? `questions-container-training-${videoIndex}` : `questions-container-${videoIndex}`;
                    const questionsSection = document.getElementById(containerID);
                    if (questionsSection && options.htmlElement) {
                        const row = options.htmlElement.closest('.sd-row');
                        if (row && row.parentNode && row.parentNode !== questionsSection) {
                            questionsSection.appendChild(row);
                        }
                    }
                }, 50);
            });
        },
        on_load: function() {
            // Only add jump/quit handlers for main task
            if (!isTraining) {
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
                                const videoTrialSet = createVideoTrial(i, false);
                                new_timeline.push(videoTrialSet);
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
                    }
                    
                    if (quitButton) {
                        quitButton.removeEventListener('click', handleQuit);
                        quitButton.addEventListener('click', handleQuit);
                    }
                    
                    if (jumpInput) {
                        jumpInput.addEventListener('keypress', function(e) {
                            if (e.key === 'Enter') {
                                handleJump();
                            }
                        });
                    }
                }, 500);
            }
        },
        on_finish: async function(data) {
            if (isTraining) {
                // Store training response
                trainingResponses.push(data.response);
                console.log('Training response stored:', trainingResponses.length);
                
                // SAVE TRAINING ANNOTATION TO DATABASE
                try {
                    const trainingAnnotationData = {
                        videoFilename: video.videoFilename,
                        description: video.description || '',
                        isTraining: true,
                        attemptNumber: Math.floor(currentTrainingAttemptStartIndex / CONFIG.TRAINING_VIDEOS_PER_ATTEMPT) + 1,
                        taskType: taskType,
                        ...data.response
                    };
                    console.log('Saving training annotation:', trainingAnnotationData);
                    await saveAnnotation(trainingAnnotationData);
                    console.log('Saved training annotation:', video.videoFilename);
                } catch (error) {
                    console.error('Failed to save training annotation:', error);
                    // Don't block the user, just log the error
                }
            } else {
                // Save main task annotation
                const trialData = data.response;
                const annotationData = {
                    videoFilename: video.videoFilename,
                    description: video.description,
                    taskType: taskType,
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
        }
    };

    return {
        timeline: [surveyTrial]
    };
}

// Run the experiment
jsPsych.run(timeline);