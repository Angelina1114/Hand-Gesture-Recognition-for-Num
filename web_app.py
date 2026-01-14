"""
手勢數字辨識 Web 應用程式（後端主模組）
==================================================

這個檔案是整個「雙手手勢數字辨識系統」的後端入口，負責：

1. 啟動 Flask Web 伺服器，提供瀏覽器可連線的 HTTP 服務
2. 使用 OpenCV 讀取攝像頭影像並透過 MJPEG 形式串流到前端
3. 呼叫 `HandDetector` 進行手部關鍵點偵測與手指狀態判斷
4. 呼叫 `GestureRecognizer` 將手指狀態轉成數字或特殊手勢
5. 實作「雙手模式」：
   - 同一畫面中最多兩隻手
   - 依照「由左到右」決定十位數與個位數
   - 兩隻手都是數字手勢 → 組成兩位數（例如左手2、右手3 → 23）
   - 若任一隻手為特殊手勢 → 使用「名稱 + 名稱」的方式表示（例如 Like+OK）
6. 提供多個 API 端點給前端使用：
   - `/video_feed`：回傳 MJPEG 影片串流（<img> 可以直接引用）
   - `/gesture_data`：回傳目前穩定辨識到的手勢結果（JSON）
   - `/gesture_help`：回傳所有支援手勢的說明文字（JSON）
   - `/camera_control`：接受「start / stop」指令以開啟或關閉攝像頭

主要資料流說明：
------------------
1. 前端透過 `<img src="/video_feed">` 取得即時影像
2. Flask 內的 `generate_frames()`：
   - 從攝像頭讀取每一幀影像
   - 交給 `HandDetector` 找出手部位置與 21 個關鍵點
   - 取得每隻手的「手指是否伸直」陣列，例如 [0,1,1,0,0]
   - 交給 `GestureRecognizer` 轉成手勢（0-9 或 Like / OK / ROCK / FUCK）
   - 若同時偵測到兩隻手，依據 X 座標由左到右排序，組合成兩位數或「手勢+手勢」
   - 套用穩定性過濾（同一結果需連續出現 N 幀才算有效）
   - 將結果寫入 `current_gesture` 全域變數
3. 前端每隔一段時間呼叫 `/gesture_data`：
   - 取得 `current_gesture`（number / name / confidence）
   - 在右側 UI 顯示對應數字或表情符號

關於 `current_gesture` 結構：
-----------------------------
    current_gesture = {
        "number": int,
        "name": str,
        "confidence": int  # 0-100
    }

- 單手數字手勢：number = 0~9,      name = "0" ~ "9"
- 雙手數字手勢：number = 10~99,    name = "10" 等
- 單一特殊手勢：number = 10~13,    name = "Like" / "OK" / ...
- 組合特殊手勢：number = -2,       name = "Like+OK" 等（左手在前，右手在後）
- 無有效手勢：  number = -1,       name = "No Hand Detected" / "Detecting..." 等提示字串

前端 `static/js/main.js` 會依照上述規則解讀並顯示對應內容。
"""
import cv2
import time
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from hand_detector import HandDetector
from gesture_recognizer import GestureRecognizer
import threading

# 創建 Flask 應用實例
app = Flask(__name__)

# ===== 全域變數 =====
# 這些變數在多個函數間共享，用於存儲系統狀態

camera = None              # OpenCV 攝像頭對象
detector = None            # 手部檢測器對象
recognizer = None          # 手勢辨識器對象

# 當前識別出的手勢（供前端查詢）
current_gesture = {
    "number": -1,          # 數字 (0-5)，-1 表示未識別
    "name": "未知",        # 中文名稱
    "confidence": 0        # 信心度 (0-100)
}

# 執行緒鎖，用於保護攝像頭資源（避免多執行緒衝突）
camera_lock = threading.Lock()

# 攝像頭運行狀態標記
is_camera_running = False

# 攝像頭啟用狀態（用戶控制）
is_camera_enabled = True

# ===== 系統配置參數 =====
CAMERA_WIDTH = 1280       # 攝像頭寬度（像素），提高解析度以改善畫質
CAMERA_HEIGHT = 720       # 攝像頭高度（像素）
CAMERA_ID = 0             # 攝像頭設備 ID（0 = 第一個攝像頭）
FPS = 30                  # 目標幀率（每秒幀數）
JPEG_QUALITY = 95         # JPEG 壓縮質量 (1-100)，95 = 高質量


def initialize_camera():
    """
    初始化攝像頭和檢測器
    
    初始化流程：
    1. 打開攝像頭設備
    2. 設定攝像頭參數（解析度、幀率）
    3. 創建手部檢測器和手勢辨識器實例
    4. 更新運行狀態
    
    返回:
        bool: True = 初始化成功, False = 初始化失敗
    """
    global camera, detector, recognizer, is_camera_running
    
    try:
        # 步驟 1: 打開攝像頭
        # cv2.VideoCapture(0) 會打開第一個可用的攝像頭
        camera = cv2.VideoCapture(CAMERA_ID)
        
        # 步驟 2: 設定攝像頭參數
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)    # 設定寬度
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)  # 設定高度
        camera.set(cv2.CAP_PROP_FPS, FPS)                     # 設定幀率
        
        # 檢查攝像頭是否成功打開
        if not camera.isOpened():
            print(f"錯誤：無法打開攝像頭 {CAMERA_ID}")
            print("請檢查：")
            print("  1. 攝像頭是否正確連接")
            print("  2. 是否有其他程式正在使用攝像頭")
            print("  3. 攝像頭權限是否正確")
            return False
        
        # 步驟 3: 創建檢測器實例
        # max_hands=2: 檢測兩隻手
        # detection_confidence=0.7: 檢測信心度閾值（0.0-1.0）
        detector = HandDetector(max_hands=2, detection_confidence=0.7)
        recognizer = GestureRecognizer()
        
        # 步驟 4: 更新狀態標記
        is_camera_running = True
        
        print(f"✅ 攝像頭初始化成功！解析度: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
        return True
        
    except Exception as e:
        # 捕捉任何異常並輸出錯誤信息
        print(f"❌ 初始化攝像頭時發生錯誤: {e}")
        return False


def generate_frames():
    """
    生成 MJPEG 影像串流（生成器函數）
    
    這是一個 Python 生成器（generator），使用 yield 關鍵字持續產生影像幀
    Flask 會自動將這些幀組合成 MJPEG 串流發送到瀏覽器
    
    工作流程：
    1. 初始化攝像頭（如果尚未初始化）
    2. 持續循環讀取影像
    3. 檢測手部並識別手勢
    4. 將結果繪製在影像上
    5. 將影像編碼為 JPEG
    6. 使用 yield 返回影像數據（不中斷循環）
    
    穩定性過濾機制：
    - 只有當相同手勢連續檢測到 N 次時，才認定為有效
    - 這樣可以避免誤判和抖動
    
    Yields:
        bytes: MJPEG 格式的影像幀數據
    """
    global current_gesture, is_camera_enabled
    
    # 確保攝像頭已初始化
    if not is_camera_running:
        if not initialize_camera():
            return  # 初始化失敗，結束生成器
    
    # ===== 穩定性過濾變數 =====
    stable_gesture = -1        # 當前穩定的手勢
    stable_count = 0           # 連續檢測到相同手勢的次數
    stable_threshold = 5       # 需要連續檢測多少次才認定為穩定（可調整）
    
    # FPS 計算變數
    previous_time = time.time()
    
    # ===== 主循環：持續處理影像 =====
    while True:
        # 檢查攝像頭是否被用戶啟用
        if not is_camera_enabled:
            # 攝像頭已關閉，生成黑色畫面
            frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
            
            # 在黑色畫面上顯示文字
            cv2.putText(
                frame,
                "CAMERA OFF",
                (CAMERA_WIDTH // 2 - 150, CAMERA_HEIGHT // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (128, 128, 128),
                2
            )
            
            # 編碼並返回
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       frame_bytes + b'\r\n')
            
            # 暫停一下再繼續
            time.sleep(0.1)
            continue
        
        # 使用執行緒鎖保護攝像頭讀取操作
        # 這樣可以避免多個執行緒同時訪問攝像頭導致的衝突
        with camera_lock:
            # 檢查攝像頭是否仍然可用
            if camera is None or not camera.isOpened():
                break  # 攝像頭不可用，退出循環
            
            # 讀取一幀影像
            # success: 是否成功讀取
            # frame: 影像數據（NumPy 陣列）
            success, frame = camera.read()
            
        # 檢查是否成功讀取影像
        if not success:
            print("⚠️ 警告：無法讀取攝像頭畫面")
            break  # 讀取失敗，退出循環
        
        # ===== 影像預處理 =====
        # 水平翻轉影像，產生鏡像效果
        # 這樣用戶看到的畫面更符合直覺（就像照鏡子）
        frame = cv2.flip(frame, 1)
        
        # ===== 手部檢測 =====
        # find_hands() 會：
        #   1. 檢測影像中的手部
        #   2. 在影像上繪製 21 個關鍵點和連接線
        #   3. 返回處理後的影像
        frame = detector.find_hands(frame, draw=True)
        
        # 獲取手部關鍵點的像素座標
        # 如果沒有檢測到手部，landmark_list 將是空列表
        landmark_list = detector.find_position(frame)
        
        # ===== 雙手手勢辨識 =====
        hand_count = detector.get_hand_count()
        
        if hand_count > 0:
            # 有檢測到手部
            hands_data = []
            
            # 遍歷所有檢測到的手
            for hand_no in range(hand_count):
                hand_landmarks = detector.find_position(frame, hand_no)
                if len(hand_landmarks) != 0:
                    # 判斷手指狀態
                    fingers = detector.fingers_up(hand_landmarks)
                    # 識別手勢
                    number, gesture_name = recognizer.recognize_number(fingers)
                    # 獲取手腕 X 座標（用於判斷左右）
                    wrist_x = hand_landmarks[0][1]
                    
                    hands_data.append({
                        'number': number,
                        'name': gesture_name,
                        'wrist_x': wrist_x
                    })
            
            # 根據 X 座標排序（由左到右）
            hands_data.sort(key=lambda h: h['wrist_x'])
            
            # 組合手勢結果
            if len(hands_data) == 1:
                # 只有一隻手
                combined_number = hands_data[0]['number']
                combined_name = hands_data[0]['name']
                
            elif len(hands_data) == 2:
                # 兩隻手
                left_hand = hands_data[0]
                right_hand = hands_data[1]
                
                # 判斷是否都是數字手勢（0-9）
                if (0 <= left_hand['number'] <= 9 and 
                    0 <= right_hand['number'] <= 9):
                    # 組成兩位數：左手是十位數，右手是個位數
                    combined_number = left_hand['number'] * 10 + right_hand['number']
                    combined_name = str(combined_number)
                else:
                    # 有特殊手勢，用 + 連接
                    combined_number = -2  # 特殊標記表示組合手勢
                    combined_name = f"{left_hand['name']}+{right_hand['name']}"
            
            else:
                combined_number = -1
                combined_name = "Unknown"
            
            # ===== 穩定性過濾機制 =====
            if combined_name == stable_gesture:
                stable_count += 1
            else:
                stable_gesture = combined_name
                stable_count = 1
            
            # 檢查手勢是否已經穩定
            if stable_count >= stable_threshold and combined_number != -1:
                # 手勢已穩定，可以顯示結果
                current_gesture = {
                    "number": combined_number,
                    "name": combined_name,
                    "confidence": min(100, int(stable_count / stable_threshold * 100))
                }
                
                # 準備要顯示的文字
                if combined_number == -2:
                    # 組合手勢
                    display_text = combined_name
                elif combined_number >= 10 and combined_number <= 99:
                    # 兩位數
                    display_text = f"Number: {combined_number}"
                elif combined_number > 99:
                    # 特殊手勢
                    display_text = combined_name
                else:
                    # 單位數
                    display_text = f"Number: {combined_number}"
                
                # ===== 在影像上繪製結果 =====
                box_width = max(350, len(display_text) * 15 + 50)
                
                # 繪製綠色背景框
                cv2.rectangle(frame, (10, 10), (box_width, 80), (0, 128, 0), -1)
                cv2.rectangle(frame, (10, 10), (box_width, 80), (255, 255, 255), 2)
                
                # 繪製文字
                cv2.putText(
                    frame,
                    display_text,
                    (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (255, 255, 255),
                    3
                )
            else:
                # 手勢尚未穩定
                current_gesture = {"number": -1, "name": "Detecting...", "confidence": 0}
                
        else:
            # 沒有檢測到手部
            stable_gesture = -1
            stable_count = 0
            current_gesture = {"number": -1, "name": "No Hand Detected", "confidence": 0}
            
            # 顯示提示文字
            cv2.putText(
                frame,
                "Place your hands in front of camera",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
        
        # ===== 計算並顯示 FPS（每秒幀數）=====
        current_time = time.time()
        
        # FPS = 1 / 時間差
        time_diff = current_time - previous_time
        fps = 1 / time_diff if time_diff > 0 else 0
        
        # 更新時間記錄
        previous_time = current_time
        
        # 在影像右上角顯示 FPS（綠色文字）
        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (CAMERA_WIDTH - 120, 30),           # 右上角位置
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),                        # 綠色
            2
        )
        
        # ===== 編碼影像為 JPEG =====
        # cv2.imencode() 將 NumPy 陣列編碼為 JPEG 格式
        # 參數說明：
        #   '.jpg': 輸出格式
        #   frame: 要編碼的影像
        #   [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]: JPEG 質量設定
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        
        # 檢查編碼是否成功
        if not ret:
            continue  # 編碼失敗，跳過這一幀
        
        # 將編碼後的數據轉換為 bytes
        frame_bytes = buffer.tobytes()
        
        # ===== 使用 yield 返回影像數據 =====
        # 這是 MJPEG 串流的標準格式
        # yield 會暫停函數執行並返回數據，但保留函數狀態
        # 下次調用時會從 yield 之後繼續執行
        yield (b'--frame\r\n'                                   # MJPEG 邊界標記
               b'Content-Type: image/jpeg\r\n\r\n' +            # HTTP 標頭
               frame_bytes + b'\r\n')                           # JPEG 數據


# ===== Flask 路由定義 =====
# 路由（Route）定義了 URL 與 Python 函數的對應關係

@app.route('/')
def index():
    """
    首頁路由
    
    URL: http://IP地址:5000/
    
    返回主頁面 HTML，包含視訊串流和手勢顯示界面
    """
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """
    視訊串流路由
    
    URL: http://IP地址:5000/video_feed
    
    返回 MJPEG 視訊串流，可以直接在 <img> 標籤中使用
    
    範例:
        <img src="http://192.168.0.154:5000/video_feed">
    
    Returns:
        Response: MJPEG 串流響應
    """
    return Response(
        generate_frames(),                              # 生成器函數
        mimetype='multipart/x-mixed-replace; boundary=frame'  # MJPEG MIME 類型
    )


@app.route('/gesture_data')
def gesture_data():
    """
    手勢數據 API 路由
    
    URL: http://IP地址:5000/gesture_data
    
    返回當前識別出的手勢數據（JSON 格式）
    前端 JavaScript 會定期調用此 API 來更新顯示
    
    返回格式:
        {
            "number": 2,           # 數字 (0-5)，-1 表示未識別
            "name": "二",          # 中文名稱
            "confidence": 100      # 信心度 (0-100)
        }
    """
    return jsonify(current_gesture)


@app.route('/gesture_help')
def gesture_help():
    """
    手勢說明 API 路由
    
    URL: http://IP地址:5000/gesture_help
    
    返回所有手勢的說明信息（JSON 陣列）
    """
    help_data = []
    # 數字 0-9
    for i in range(10):
        help_data.append({
            "id": i,
            "type": "number",
            "description": recognizer.get_gesture_description(i)
        })
    # 特殊手勢
    special_gestures = [
        (10, "Like 👍"),
        (11, "OK 👌"),
        (12, "ROCK 🤘"),
        (13, "FUCK 🖕")
    ]
    for gesture_id, name in special_gestures:
        help_data.append({
            "id": gesture_id,
            "type": "special",
            "name": name,
            "description": recognizer.get_gesture_description(gesture_id)
        })
    return jsonify(help_data)


@app.route('/camera_control', methods=['POST'])
def camera_control():
    """
    攝像頭控制 API 路由
    
    URL: http://IP地址:5000/camera_control
    Method: POST
    
    接收控制指令，啟動或停止攝像頭
    
    請求格式:
        {
            "action": "start" 或 "stop"
        }
    
    返回格式:
        {
            "status": "success" 或 "error",
            "message": "說明信息",
            "camera_enabled": true/false
        }
    """
    global is_camera_enabled
    
    try:
        # 獲取請求數據
        data = request.get_json()
        action = data.get('action', '')
        
        if action == 'start':
            # 啟動攝像頭
            is_camera_enabled = True
            return jsonify({
                "status": "success",
                "message": "Camera started",
                "camera_enabled": True
            })
        
        elif action == 'stop':
            # 停止攝像頭
            is_camera_enabled = False
            # 清除當前手勢狀態
            global current_gesture
            current_gesture = {
                "number": -1,
                "name": "Camera Off",
                "confidence": 0
            }
            return jsonify({
                "status": "success",
                "message": "Camera stopped",
                "camera_enabled": False
            })
        
        else:
            return jsonify({
                "status": "error",
                "message": "Invalid action. Use 'start' or 'stop'.",
                "camera_enabled": is_camera_enabled
            }), 400
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "camera_enabled": is_camera_enabled
        }), 500


def cleanup():
    """
    清理系統資源
    
    在程式結束前調用，確保：
    1. 攝像頭被正確釋放
    2. 沒有資源洩漏
    """
    global camera, is_camera_running
    
    # 更新狀態標記
    is_camera_running = False
    
    # 釋放攝像頭資源
    if camera is not None:
        with camera_lock:
            camera.release()
        print("📷 攝像頭已關閉")


# ===== 主程式入口 =====
if __name__ == '__main__':
    # 只有直接執行此腳本時才會執行以下代碼
    # 如果被其他模組導入，則不會執行
    
    try:
        # 顯示啟動信息
        print("=" * 60)
        print("🤚 手勢數字辨識 Web 系統")
        print("=" * 60)
        print("正在啟動伺服器...")
        print(f"請在瀏覽器中訪問: http://<Jetson的IP地址>:5000")
        print(f"或在本機訪問: http://localhost:5000")
        print("按 Ctrl+C 停止伺服器")
        print("=" * 60)
        
        # ===== 啟動 Flask 開發伺服器 =====
        # 參數說明：
        #   host='0.0.0.0': 監聽所有網路介面，允許外部設備訪問
        #                   如果設為 '127.0.0.1' 則只能本機訪問
        #   port=5000: HTTP 伺服器端口號
        #   debug=False: 不啟用除錯模式（生產環境應關閉）
        #   threaded=True: 使用多執行緒處理請求（支援並發連接）
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        # 用戶按下 Ctrl+C 中斷程式
        print("\n\n⏹️  正在關閉伺服器...")
        
    except Exception as e:
        # 捕捉其他異常
        print(f"\n❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()  # 輸出完整的錯誤堆疊
        
    finally:
        # 無論如何都會執行的清理代碼
        # 確保資源被正確釋放
        cleanup()

