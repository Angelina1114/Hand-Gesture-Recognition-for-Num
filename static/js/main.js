/**
 * Hand Gesture Recognition System - JavaScript
 * Handles real-time gesture data updates and UI interactions
 */

// Configuration
const CONFIG = {
    updateInterval: 200,  // Update gesture data every 200ms
    apiEndpoint: '/gesture_data',
    cameraControlEndpoint: '/camera_control',
    specialGestures: {
        10: '👍',  // Like
        11: '👌',  // OK
        12: '🤘',  // ROCK
        13: '🖕'   // FUCK
    }
};

// DOM Elements
const elements = {
    numberDisplay: null,
    nameDisplay: null,
    confidenceFill: null,
    videoStream: null,
    cameraToggleBtn: null,
    cameraOffOverlay: null
};

// State
let isCameraOn = true;

/**
 * Initialize the application
 */
function init() {
    // Get DOM elements
    elements.numberDisplay = document.getElementById('gestureNumber');
    elements.nameDisplay = document.getElementById('gestureName');
    elements.confidenceFill = document.getElementById('confidenceFill');
    elements.videoStream = document.getElementById('videoStream');
    elements.cameraToggleBtn = document.getElementById('cameraToggle');
    elements.cameraOffOverlay = document.getElementById('cameraOffOverlay');
    
    // Set up event listeners
    setupEventListeners();
    
    // Start gesture data polling
    startGesturePolling();
    
    console.log('Hand Gesture Recognition System initialized');
}

/**
 * Set up event listeners
 */
function setupEventListeners() {
    // Handle video stream errors
    if (elements.videoStream) {
        elements.videoStream.onerror = handleVideoError;
    }
    
    // Handle camera toggle button
    if (elements.cameraToggleBtn) {
        elements.cameraToggleBtn.addEventListener('click', toggleCamera);
    }
}

/**
 * Handle video stream errors
 */
function handleVideoError() {
    console.error('Video stream error');
    if (elements.videoStream) {
        elements.videoStream.alt = 'Unable to load video stream. Please check camera connection.';
    }
}

/**
 * Start polling gesture data from the server
 */
function startGesturePolling() {
    setInterval(updateGestureData, CONFIG.updateInterval);
}

/**
 * Fetch and update gesture data
 */
async function updateGestureData() {
    try {
        const response = await fetch(CONFIG.apiEndpoint);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        updateUI(data);
        
    } catch (error) {
        console.error('Error fetching gesture data:', error);
        showError();
    }
}

/**
 * Update UI with gesture data
 * @param {Object} data - Gesture data from server
 */
function updateUI(data) {
    if (!elements.numberDisplay || !elements.nameDisplay || !elements.confidenceFill) {
        console.error('UI elements not found');
        return;
    }
    
    if (data.number >= 0) {
        // Valid gesture detected
        updateGestureDisplay(data);
        showConfidence(data.confidence);
    } else {
        // No gesture or unknown gesture
        showNoDetection(data.name);
    }
}

/**
 * Update gesture display
 * @param {Object} data - Gesture data
 */
function updateGestureDisplay(data) {
    const { number, name } = data;
    
    // Update number/icon display
    if (number === -2) {
        // Combined gestures (e.g., "Like+OK")
        // Parse and display combined emojis or text
        elements.numberDisplay.textContent = parseGestureName(name);
        elements.numberDisplay.style.fontSize = '3em';  // Smaller for combined
    } else if (number >= 10 && number <= 99) {
        // Two-digit number
        elements.numberDisplay.textContent = number;
        elements.numberDisplay.style.fontSize = '5em';
    } else if (number > 99) {
        // Single special gesture - show emoji
        const emoji = CONFIG.specialGestures[number] || '?';
        elements.numberDisplay.textContent = emoji;
        elements.numberDisplay.style.fontSize = '5em';
    } else {
        // Single digit number
        elements.numberDisplay.textContent = number;
        elements.numberDisplay.style.fontSize = '5em';
    }
    
    // Update name display
    elements.nameDisplay.textContent = name;
    elements.nameDisplay.classList.remove('no-detection');
}

/**
 * Parse gesture name to display format
 * @param {string} name - Gesture name (e.g., "Like+OK")
 * @returns {string} Display text with emojis
 */
function parseGestureName(name) {
    if (!name || !name.includes('+')) {
        return name;
    }
    
    // Split by + and convert each part to emoji if possible
    const parts = name.split('+');
    let result = '';
    
    for (let i = 0; i < parts.length; i++) {
        const part = parts[i].trim();
        
        // Try to find emoji for special gestures
        let display = part;
        if (part === 'Like') display = '👍';
        else if (part === 'OK') display = '👌';
        else if (part === 'ROCK') display = '🤘';
        else if (part === 'FUCK') display = '🖕';
        else if (!isNaN(part)) display = part;  // Keep numbers as is
        
        result += display;
        if (i < parts.length - 1) {
            result += '+';
        }
    }
    
    return result;
}

/**
 * Show confidence level
 * @param {number} confidence - Confidence percentage (0-100)
 */
function showConfidence(confidence) {
    elements.confidenceFill.style.width = `${confidence}%`;
}

/**
 * Show no detection state
 * @param {string} message - Message to display
 */
function showNoDetection(message) {
    elements.numberDisplay.textContent = '?';
    elements.nameDisplay.textContent = message || 'Waiting...';
    elements.nameDisplay.classList.add('no-detection');
    elements.confidenceFill.style.width = '0%';
}

/**
 * Show error state
 */
function showError() {
    showNoDetection('Connection Error');
}

/**
 * Toggle camera on/off
 */
async function toggleCamera() {
    try {
        // Send request to backend
        const response = await fetch(CONFIG.cameraControlEndpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                action: isCameraOn ? 'stop' : 'start'
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.status === 'success') {
            isCameraOn = !isCameraOn;
            updateCameraUI();
        } else {
            console.error('Camera control failed:', data.message);
        }
        
    } catch (error) {
        console.error('Error controlling camera:', error);
    }
}

/**
 * Update camera UI based on state
 */
function updateCameraUI() {
    if (!elements.cameraToggleBtn || !elements.cameraOffOverlay) return;
    
    const btnIcon = elements.cameraToggleBtn.querySelector('.btn-icon');
    const btnText = elements.cameraToggleBtn.querySelector('.btn-text');
    
    if (isCameraOn) {
        // Camera is ON
        elements.cameraToggleBtn.classList.remove('camera-off');
        elements.cameraOffOverlay.style.display = 'none';
        if (btnIcon) btnIcon.textContent = '●';
        if (btnText) btnText.textContent = 'STOP CAMERA';
        
        // Reload video stream
        const timestamp = new Date().getTime();
        const videoFeedUrl = elements.videoStream.getAttribute('data-video-url') || '/video_feed';
        elements.videoStream.src = `${videoFeedUrl}?t=${timestamp}`;
    } else {
        // Camera is OFF
        elements.cameraToggleBtn.classList.add('camera-off');
        elements.cameraOffOverlay.style.display = 'flex';
        if (btnIcon) btnIcon.textContent = '○';
        if (btnText) btnText.textContent = 'START CAMERA';
        
        // Clear video stream
        elements.videoStream.src = '';
        
        // Show camera off message
        showNoDetection('Camera Off');
    }
}

/**
 * Utility: Format gesture name
 * @param {string} name - Raw gesture name
 * @returns {string} Formatted name
 */
function formatGestureName(name) {
    if (!name) return 'Unknown';
    return name.charAt(0).toUpperCase() + name.slice(1);
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

// Export for potential use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        init,
        updateGestureData,
        CONFIG
    };
}

