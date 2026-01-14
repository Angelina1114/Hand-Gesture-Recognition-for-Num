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
    
    # 初始化手部檢測器和手勢辨識器（雙手模式）
    detector = HandDetector(max_hands=2, detection_confidence=0.7)
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
        hand_count = detector.get_hand_count()
        
        # 雙手辨識手勢
        if hand_count > 0:
            hands_data = []
            
            # 遍歷所有檢測到的手
            for hand_no in range(hand_count):
                hand_landmarks = detector.find_position(img, hand_no)
                if len(hand_landmarks) != 0:
                    fingers = detector.fingers_up(hand_landmarks)
                    number, gesture_name = recognizer.recognize_number(fingers)
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
                combined_number = hands_data[0]['number']
                combined_name = hands_data[0]['name']
            elif len(hands_data) == 2:
                left_hand = hands_data[0]
                right_hand = hands_data[1]
                
                if (0 <= left_hand['number'] <= 9 and 
                    0 <= right_hand['number'] <= 9):
                    # 組成兩位數
                    combined_number = left_hand['number'] * 10 + right_hand['number']
                    combined_name = str(combined_number)
                else:
                    # 組合手勢
                    combined_number = -2
                    combined_name = f"{left_hand['name']}+{right_hand['name']}"
            else:
                combined_number = -1
                combined_name = "Unknown"
            
            # 穩定性檢測
            if combined_name == stable_gesture:
                stable_count += 1
            else:
                stable_gesture = combined_name
                stable_count = 1
            
            # 如果手勢穩定，則顯示
            if stable_count >= stable_threshold and combined_number != -1:
                # 準備顯示文字
                if combined_number == -2:
                    display_text = combined_name
                elif 10 <= combined_number <= 99:
                    display_text = f"Number: {combined_number}"
                elif combined_number > 99:
                    display_text = combined_name
                else:
                    display_text = f"Number: {combined_number}"
                
                # 動態調整背景框寬度
                box_width = max(350, len(display_text) * 15 + 50)
                
                # 背景框
                cv2.rectangle(img, (10, 10), (box_width, 80), (0, 128, 0), -1)
                cv2.rectangle(img, (10, 10), (box_width, 80), (255, 255, 255), 2)
                
                # 顯示文字
                cv2.putText(img, display_text, (20, 55), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                
                # 在終端輸出
                print(f"\r識別結果: {display_text}", end="", flush=True)
        else:
            # 沒有檢測到手部
            stable_gesture = -1
            stable_count = 0
            cv2.putText(img, "Place your hands in front of camera", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 計算並顯示 FPS
        current_time = time.time()
        fps = 1 / (current_time - previous_time) if (current_time - previous_time) > 0 else 0
        previous_time = current_time
        
        cv2.putText(img, f"FPS: {int(fps)}", (camera_width - 120, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 顯示說明信息（英文）
        cv2.putText(img, "Press 'q' or 'ESC' to quit | 'h' for help", 
                   (10, camera_height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 顯示影像
        cv2.imshow("Hand Gesture Recognition System", img)
        
        # 鍵盤輸入處理
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # 'q' 或 ESC
            print("\n\n程式退出")
            break
        elif key == ord('h'):  # 幫助
            print("\n" + "=" * 60)
            print("手勢辨識系統 - 幫助信息")
            print("=" * 60)
            print("【數字 0-9】")
            for i in range(10):
                print(f"  {i}: {recognizer.get_gesture_description(i)}")
            print("\n【特殊手勢】")
            print(f"  👍 Like: {recognizer.get_gesture_description(10)}")
            print(f"  👌 OK: {recognizer.get_gesture_description(11)}")
            print(f"  🤘 ROCK: {recognizer.get_gesture_description(12)}")
            print(f"  🖕 FUCK: {recognizer.get_gesture_description(13)}")
            print("=" * 60 + "\n")
    
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

