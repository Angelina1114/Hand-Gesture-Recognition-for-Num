"""
手勢數字辨識主程式
使用 Logitech C270 攝像頭進行實時手勢辨識
"""
import cv2
import time
from hand_detector import HandDetector
from gesture_recognizer import GestureRecognizer


def main():
    # 設定參數
    camera_width = 640
    camera_height = 480
    camera_id = 0  # 通常是 0，如果有多個攝像頭可以嘗試 1, 2...
    
    # 初始化攝像頭
    print("正在初始化攝像頭...")
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    
    if not cap.isOpened():
        print(f"錯誤：無法打開攝像頭 {camera_id}")
        print("請確認：")
        print("1. 攝像頭已正確連接")
        print("2. 攝像頭驅動已安裝")
        print("3. 您有訪問攝像頭的權限")
        return
    
    print(f"攝像頭初始化成功！解析度: {camera_width}x{camera_height}")
    
    # 初始化手部檢測器和手勢辨識器
    detector = HandDetector(max_hands=1, detection_confidence=0.7)
    recognizer = GestureRecognizer()
    
    # FPS 計算
    previous_time = 0
    
    # 穩定性計數器（避免誤判）
    stable_gesture = -1
    stable_count = 0
    stable_threshold = 5  # 需要連續檢測到相同手勢 5 次才顯示
    
    print("\n開始辨識...")
    print("按 'q' 或 'ESC' 退出程式")
    print("按 'h' 顯示幫助信息")
    print("-" * 50)
    
    while True:
        success, img = cap.read()
        
        if not success:
            print("警告：無法讀取攝像頭畫面")
            break
        
        # 水平翻轉影像（鏡像效果）
        img = cv2.flip(img, 1)
        
        # 檢測手部
        img = detector.find_hands(img, draw=True)
        landmark_list = detector.find_position(img)
        
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
                # 繪製結果
                display_text = f"數字: {number} ({gesture_name})"
                
                # 背景框
                cv2.rectangle(img, (10, 10), (400, 80), (0, 128, 0), -1)
                cv2.rectangle(img, (10, 10), (400, 80), (255, 255, 255), 2)
                
                # 顯示數字
                cv2.putText(img, display_text, (20, 55), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                
                # 在終端輸出
                print(f"\r識別結果: {display_text}", end="", flush=True)
        else:
            # 沒有檢測到手部
            stable_gesture = -1
            stable_count = 0
            cv2.putText(img, "請將手放在攝像頭前", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # 計算並顯示 FPS
        current_time = time.time()
        fps = 1 / (current_time - previous_time) if (current_time - previous_time) > 0 else 0
        previous_time = current_time
        
        cv2.putText(img, f"FPS: {int(fps)}", (camera_width - 120, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 顯示說明信息
        cv2.putText(img, "按 'q' 或 'ESC' 退出 | 按 'h' 幫助", 
                   (10, camera_height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 顯示影像
        cv2.imshow("手勢數字辨識系統", img)
        
        # 鍵盤輸入處理
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # 'q' 或 ESC
            print("\n\n程式退出")
            break
        elif key == ord('h'):  # 幫助
            print("\n" + "=" * 50)
            print("手勢數字辨識系統 - 幫助信息")
            print("=" * 50)
            for i in range(6):
                print(f"{i}: {recognizer.get_gesture_description(i)}")
            print("=" * 50 + "\n")
    
    # 清理資源
    cap.release()
    cv2.destroyAllWindows()
    print("攝像頭已關閉")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程式被中斷")
    except Exception as e:
        print(f"\n錯誤: {e}")
        import traceback
        traceback.print_exc()

