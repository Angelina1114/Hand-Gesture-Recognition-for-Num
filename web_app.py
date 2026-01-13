"""
手勢數字辨識 Web 應用程式
使用 Flask 提供網頁介面，可在瀏覽器中查看實時辨識結果
"""
import cv2
import time
from flask import Flask, render_template, Response, jsonify
from hand_detector import HandDetector
from gesture_recognizer import GestureRecognizer
import threading

app = Flask(__name__)

# 全域變數
camera = None
detector = None
recognizer = None
current_gesture = {"number": -1, "name": "未知", "confidence": 0}
camera_lock = threading.Lock()
is_camera_running = False

# 設定參數
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_ID = 0
FPS = 30


def initialize_camera():
    """初始化攝像頭"""
    global camera, detector, recognizer, is_camera_running
    
    try:
        camera = cv2.VideoCapture(CAMERA_ID)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        camera.set(cv2.CAP_PROP_FPS, FPS)
        
        if not camera.isOpened():
            print(f"錯誤：無法打開攝像頭 {CAMERA_ID}")
            return False
        
        detector = HandDetector(max_hands=1, detection_confidence=0.7)
        recognizer = GestureRecognizer()
        is_camera_running = True
        
        print(f"攝像頭初始化成功！解析度: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
        return True
    except Exception as e:
        print(f"初始化攝像頭時發生錯誤: {e}")
        return False


def generate_frames():
    """產生影像串流"""
    global current_gesture
    
    if not is_camera_running:
        if not initialize_camera():
            return
    
    stable_gesture = -1
    stable_count = 0
    stable_threshold = 5
    previous_time = time.time()
    
    while True:
        with camera_lock:
            if camera is None or not camera.isOpened():
                break
            
            success, frame = camera.read()
            
        if not success:
            print("警告：無法讀取攝像頭畫面")
            break
        
        # 水平翻轉影像（鏡像效果）
        frame = cv2.flip(frame, 1)
        
        # 檢測手部
        frame = detector.find_hands(frame, draw=True)
        landmark_list = detector.find_position(frame)
        
        # 辨識手勢
        if len(landmark_list) != 0:
            fingers = detector.fingers_up(landmark_list)
            number, gesture_name = recognizer.recognize_number(fingers)
            
            # 穩定性檢測
            if number == stable_gesture:
                stable_count += 1
            else:
                stable_gesture = number
                stable_count = 1
            
            # 如果手勢穩定，則顯示
            if stable_count >= stable_threshold and number != -1:
                current_gesture = {
                    "number": number,
                    "name": gesture_name,
                    "confidence": min(100, int(stable_count / stable_threshold * 100))
                }
                
                display_text = f"數字: {number} ({gesture_name})"
                
                # 背景框
                cv2.rectangle(frame, (10, 10), (400, 80), (0, 128, 0), -1)
                cv2.rectangle(frame, (10, 10), (400, 80), (255, 255, 255), 2)
                
                # 顯示數字
                cv2.putText(frame, display_text, (20, 55), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            else:
                current_gesture = {"number": -1, "name": "偵測中...", "confidence": 0}
        else:
            stable_gesture = -1
            stable_count = 0
            current_gesture = {"number": -1, "name": "未偵測到手部", "confidence": 0}
            cv2.putText(frame, "請將手放在鏡頭前", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # 計算並顯示 FPS
        current_time = time.time()
        fps = 1 / (current_time - previous_time) if (current_time - previous_time) > 0 else 0
        previous_time = current_time
        
        cv2.putText(frame, f"FPS: {int(fps)}", (CAMERA_WIDTH - 120, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 編碼影像為 JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        
        # 產生串流
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    """首頁"""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """影像串流路由"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/gesture_data')
def gesture_data():
    """獲取當前手勢數據（JSON API）"""
    return jsonify(current_gesture)


@app.route('/gesture_help')
def gesture_help():
    """獲取手勢說明"""
    help_data = []
    for i in range(6):
        help_data.append({
            "number": i,
            "description": recognizer.get_gesture_description(i)
        })
    return jsonify(help_data)


def cleanup():
    """清理資源"""
    global camera, is_camera_running
    is_camera_running = False
    if camera is not None:
        with camera_lock:
            camera.release()
        print("攝像頭已關閉")


if __name__ == '__main__':
    try:
        print("=" * 60)
        print("手勢數字辨識 Web 系統")
        print("=" * 60)
        print("正在啟動伺服器...")
        print(f"請在瀏覽器中訪問: http://<Jetson的IP地址>:5000")
        print(f"或在本機訪問: http://localhost:5000")
        print("按 Ctrl+C 停止伺服器")
        print("=" * 60)
        
        # 啟動 Flask 伺服器
        # host='0.0.0.0' 允許外部訪問
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        print("\n\n正在關閉伺服器...")
    except Exception as e:
        print(f"\n錯誤: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup()

