"""
手勢數字辨識 Web 應用程式
使用 Flask 框架提供網頁介面，可在瀏覽器中查看實時手勢辨識結果

主要功能：
1. 透過 HTTP 提供網頁服務（支援遠端訪問）
2. 實時視訊串流（MJPEG 格式）
3. 手勢辨識數據 API（JSON 格式）
4. 響應式網頁界面

技術架構：
- 後端：Flask (Python 網頁框架)
- 視訊處理：OpenCV
- 手部檢測：MediaPipe
- 串流協議：MJPEG (Motion JPEG)
- 前端：HTML5 + CSS3 + JavaScript

工作流程：
1. Flask 啟動 HTTP 伺服器（端口 5000）
2. 攝像頭持續捕捉影像
3. MediaPipe 檢測手部並識別手勢
4. 影像編碼為 JPEG 並串流到瀏覽器
5. JavaScript 定期獲取最新的手勢數據並更新界面
"""
import cv2
import time
from flask import Flask, render_template, Response, jsonify
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
        # max_hands=1: 只檢測一隻手（提高性能）
        # detection_confidence=0.7: 檢測信心度閾值（0.0-1.0）
        detector = HandDetector(max_hands=1, detection_confidence=0.7)
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
    global current_gesture
    
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
        
        # ===== 手勢辨識 =====
        if len(landmark_list) != 0:
            # 有檢測到手部
            # 步驟 1: 判斷每根手指是否伸直
            # 返回格式: [大拇指, 食指, 中指, 無名指, 小指]
            # 例如: [0, 1, 1, 0, 0] 表示食指和中指伸直
            fingers = detector.fingers_up(landmark_list)
            
            # 步驟 2: 根據手指狀態識別數字
            # number: 0-5 的數字，-1 表示無法識別
            # gesture_name: 中文名稱（"零", "一", "二"...）
            number, gesture_name = recognizer.recognize_number(fingers)
            
            # ===== 穩定性過濾機制 =====
            # 只有當相同手勢連續檢測到多次，才認定為有效
            
            if number == stable_gesture:
                # 檢測到的手勢與上一幀相同，計數器加 1
                stable_count += 1
            else:
                # 檢測到的手勢變了，重置計數器
                stable_gesture = number
                stable_count = 1
            
            # 檢查手勢是否已經穩定
            if stable_count >= stable_threshold and number != -1:
                # 手勢已穩定，可以顯示結果
                # 更新全域變數（供前端 API 查詢）
                current_gesture = {
                    "number": number,
                    "name": gesture_name,
                    # 信心度計算：超過閾值後，每多檢測一幀增加一些信心度
                    "confidence": min(100, int(stable_count / stable_threshold * 100))
                }
                
                # 準備要顯示的文字（使用英文避免顯示問題）
                if number >= 10:
                    # 特殊手勢：直接顯示名稱
                    display_text = gesture_name
                else:
                    # 數字手勢：顯示數字
                    display_text = f"Number: {number}"
                
                # ===== 在影像上繪製結果 =====
                
                # 繪製綠色背景框（填滿）
                cv2.rectangle(frame, (10, 10), (350, 80), (0, 128, 0), -1)
                
                # 繪製白色邊框
                cv2.rectangle(frame, (10, 10), (350, 80), (255, 255, 255), 2)
                
                # 繪製文字（白色、粗體）
                cv2.putText(
                    frame,                          # 目標影像
                    display_text,                   # 要顯示的文字（英文）
                    (20, 55),                       # 文字位置（左下角座標）
                    cv2.FONT_HERSHEY_SIMPLEX,       # 字體
                    1.5,                            # 字體大小
                    (255, 255, 255),                # 顏色（白色，BGR 格式）
                    3                               # 線條粗細
                )
            else:
                # 手勢尚未穩定，顯示"偵測中"
                current_gesture = {"number": -1, "name": "偵測中...", "confidence": 0}
                
        else:
            # 沒有檢測到手部
            
            # 重置穩定性計數器
            stable_gesture = -1
            stable_count = 0
            
            # 更新狀態為"未偵測到手部"
            current_gesture = {"number": -1, "name": "未偵測到手部", "confidence": 0}
            
            # 在影像上顯示提示文字（紅色，使用英文）
            cv2.putText(
                frame,
                "Place your hand in front of camera",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),                    # 紅色（BGR 格式）
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

