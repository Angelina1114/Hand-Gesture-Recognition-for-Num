/**
 * 手勢數字辨識系統 - 前端主程式
 *
 * 這個檔案負責：
 * 1. 週期性從後端取得目前辨識到的手勢資料（/gesture_data API）
 * 2. 根據回傳的資料，更新畫面右側的大數字／表情以及文字說明
 * 3. 控制攝像頭的啟動與關閉（/camera_control API）
 * 4. 處理錯誤狀況（例如連線失敗、攝像頭關閉）並在畫面上給出提示
 *
 * 注意：
 * - 「數字手勢」：後端會傳回 number 為 0-9 或 10-99（雙手組合成兩位數）
 * - 「組合特殊手勢」：例如「Like+OK」，後端會傳回 number = -2，name = "Like+OK"
 * - 「一般非有效手勢」：number = -1，name 會是提示字串（例如 "No Hand Detected"）
 */

// ===== 全域設定 =====
const CONFIG = {
    // 從後端更新手勢資料的時間間隔（毫秒）
    updateInterval: 200,
    // 取得目前手勢資料的 API
    apiEndpoint: '/gesture_data',
    // 控制攝像頭啟動／關閉的 API
    cameraControlEndpoint: '/camera_control',
    // 特殊手勢代號對應的表情符號（只在單一特殊手勢時使用，組合手勢用名稱轉換）
    specialGestures: {
        10: '👍',  // Like（讚）
        11: '👌',  // OK
        12: '🤘',  // ROCK
        13: '🖕'   // FUCK
    }
};

// ===== 頁面上會用到的 DOM 元素 =====
const elements = {
    numberDisplay: null,   // 顯示大數字或表情符號的區塊（中央大字）
    nameDisplay: null,     // 顯示手勢名稱或描述的小字
    confidenceFill: null,  // 信心度進度條（綠色長條）
    videoStream: null,     // 影片串流 <img> 元素
    cameraToggleBtn: null, // 開啟／關閉攝像頭的按鈕
    cameraOffOverlay: null // 攝像頭關閉時覆蓋在影像上的「CAMERA OFF」圖層
};

// ===== 前端狀態 =====
// true 代表攝像頭目前啟用中，false 代表已關閉（按鈕與畫面會依此更新）
let isCameraOn = true;

/**
 * 初始化整個前端程式
 * - 取得必要的 DOM 元素
 * - 綁定事件處理（按鈕、錯誤處理等）
 * - 啟動定時向後端拉取手勢資料的機制
 */
function init() {
    // 取得頁面上的元素參照
    elements.numberDisplay = document.getElementById('gestureNumber');
    elements.nameDisplay = document.getElementById('gestureName');
    elements.confidenceFill = document.getElementById('confidenceFill');
    elements.videoStream = document.getElementById('videoStream');
    elements.cameraToggleBtn = document.getElementById('cameraToggle');
    elements.cameraOffOverlay = document.getElementById('cameraOffOverlay');
    
    // 綁定各種事件（攝像頭按鈕、影片錯誤等）
    setupEventListeners();
    
    // 啟動週期性更新手勢資料的計時器
    startGesturePolling();
    
    console.log('Hand Gesture Recognition System initialized');
}

/**
 * 綁定所有需要的事件監聽器
 * - 影片載入錯誤處理
 * - 攝像頭開關按鈕點擊事件
 */
function setupEventListeners() {
    // 影片流錯誤處理：顯示提示文字
    if (elements.videoStream) {
        elements.videoStream.onerror = handleVideoError;
    }
    
    // 攝像頭開關按鈕
    if (elements.cameraToggleBtn) {
        elements.cameraToggleBtn.addEventListener('click', toggleCamera);
    }
}

/**
 * 處理影片串流錯誤
 * 例如攝像頭沒有接上、URL 錯誤等
 */
function handleVideoError() {
    console.error('Video stream error');
    if (elements.videoStream) {
        elements.videoStream.alt = 'Unable to load video stream. Please check camera connection.';
    }
}

/**
 * 啟動「從後端定期拉取手勢資料」的機制
 */
function startGesturePolling() {
    setInterval(updateGestureData, CONFIG.updateInterval);
}

/**
 * 向後端取得最新的手勢資料，並更新畫面
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
 * 根據從後端取得的手勢資料更新畫面
 * @param {Object} data - 後端回傳的 JSON，格式例如：
 *   {
 *     number: 23,          // 或 -2（代表組合特殊手勢）, 或 -1（無手勢）
 *     name: "23" 或 "Like+OK",
 *     confidence: 0-100
 *   }
 */
function updateUI(data) {
    if (!elements.numberDisplay || !elements.nameDisplay || !elements.confidenceFill) {
        console.error('UI elements not found');
        return;
    }
    
    // data.number === -2 代表「組合特殊手勢」（例如 Like+OK），也視為有效結果
    if (data.number >= 0 || data.number === -2) {
        // 有辨識到有效手勢
        updateGestureDisplay(data);
        showConfidence(data.confidence);
    } else {
        // 沒有偵測到有效手勢（或是未知手勢）
        showNoDetection(data.name);
    }
}

/**
 * 根據手勢資料更新右側的顯示區（大字／表情 + 名稱）
 * @param {Object} data - 後端提供的手勢資料
 */
function updateGestureDisplay(data) {
    const { number, name } = data;
    
    // ===== 更新大數字或表情符號 =====
    if (number === -2) {
        // 組合手勢（例如 "Like+OK"）
        // 將文字名稱轉成對應的 emoji 或文字顯示
        elements.numberDisplay.textContent = parseGestureName(name);
        elements.numberDisplay.style.fontSize = '3em';  // Smaller for combined
    } else if (number >= 10 && number <= 99) {
        // 兩位數數字（左手十位、右手個位）
        elements.numberDisplay.textContent = number;
        elements.numberDisplay.style.fontSize = '5em';
    } else if (number > 99) {
        // 單一特殊手勢（理論上目前不會 >99，保留擴充）
        const emoji = CONFIG.specialGestures[number] || '?';
        elements.numberDisplay.textContent = emoji;
        elements.numberDisplay.style.fontSize = '5em';
    } else {
        // 單一數字（0-9）
        elements.numberDisplay.textContent = number;
        elements.numberDisplay.style.fontSize = '5em';
    }
    
    // ===== 更新手勢名稱文字 =====
    elements.nameDisplay.textContent = name;
    elements.nameDisplay.classList.remove('no-detection');
}

/**
 * 將手勢名稱（例如 "Like+OK"）轉成要顯示的大字內容
 *
 * 規則：
 * - 以 '+' 拆成多個手勢名稱
 * - 個別轉成表情符號（Like → 👍 等）
 * - 中間目前用「+」串起來（如果想改成空白／其它符號，可調整這裡）
 *
 * @param {string} name - 手勢名稱，例如 "Like+OK"
 * @returns {string} 要顯示在畫面上的字串（可能包含 emoji）
 */
function parseGestureName(name) {
    if (!name || !name.includes('+')) {
        return name;
    }
    
    // Split by + and convert each part to emoji if possible
    const parts = name.split('+');
    let result = '';
    
    // 逐一處理每一個手勢名稱
    for (let i = 0; i < parts.length; i++) {
        const part = parts[i].trim();
        
        // 嘗試將特定文字轉成對應的表情符號
        let display = part;
        if (part === 'Like') display = '👍';
        else if (part === 'OK') display = '👌';
        else if (part === 'ROCK') display = '🤘';
        else if (part === 'FUCK') display = '🖕';
        else if (!isNaN(part)) display = part;  // 如果是數字字串，直接保留
        
        result += display;
        // 中間連接符號，目前使用 '+'，你可以改成 ' ' 或 ' | ' 等樣式
        if (i < parts.length - 1) {
            result += '+';
        }
    }
    
    return result;
}

/**
 * 顯示手勢辨識的信心度（右側綠色進度條）
 * @param {number} confidence - 信心度百分比 (0-100)
 */
function showConfidence(confidence) {
    elements.confidenceFill.style.width = `${confidence}%`;
}

/**
 * 當沒有偵測到有效手勢時，更新畫面顯示「?」與提示文字
 * @param {string} message - 要顯示的提示訊息
 */
function showNoDetection(message) {
    elements.numberDisplay.textContent = '?';
    elements.nameDisplay.textContent = message || 'Waiting...';
    elements.nameDisplay.classList.add('no-detection');
    elements.confidenceFill.style.width = '0%';
}

/**
 * 當與後端溝通發生錯誤時（例如 HTTP 失敗），顯示錯誤狀態
 */
function showError() {
    showNoDetection('Connection Error');
}

/**
 * 切換攝像頭的狀態（開啟／關閉）
 *
 * 流程：
 * 1. 將 action: 'start' 或 'stop' POST 給 /camera_control
 * 2. 後端更新狀態並回傳結果
 * 3. 前端依據結果更新 UI（按鈕文字、覆蓋層、串流 URL 等）
 */
async function toggleCamera() {
    try {
        // 發送請求到後端，要求啟動或停止攝像頭
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
            // 狀態切換成功，更新前端狀態與畫面
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
 * 根據當前攝像頭狀態更新畫面：
 * - 按鈕樣式與文字（STOP / START）
 * - 是否顯示「CAMERA OFF」覆蓋層
 * - 是否載入／清除影片串流來源
 */
function updateCameraUI() {
    if (!elements.cameraToggleBtn || !elements.cameraOffOverlay) return;
    
    const btnIcon = elements.cameraToggleBtn.querySelector('.btn-icon');
    const btnText = elements.cameraToggleBtn.querySelector('.btn-text');
    
    if (isCameraOn) {
        // 攝像頭啟用中
        elements.cameraToggleBtn.classList.remove('camera-off');
        elements.cameraOffOverlay.style.display = 'none';
        if (btnIcon) btnIcon.textContent = '●';
        if (btnText) btnText.textContent = 'STOP CAMERA';
        
        // Reload video stream
        const timestamp = new Date().getTime();
        const videoFeedUrl = elements.videoStream.getAttribute('data-video-url') || '/video_feed';
        elements.videoStream.src = `${videoFeedUrl}?t=${timestamp}`;
    } else {
        // 攝像頭已關閉
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
 * 工具函式：將手勢名稱的第一個字母變大寫（目前未直接使用，保留以供擴充）
 * @param {string} name - 原始手勢名稱
 * @returns {string} 處理後的名稱
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

