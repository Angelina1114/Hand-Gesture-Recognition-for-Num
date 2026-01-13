"""
手部檢測模組
使用 Google MediaPipe 進行手部關鍵點檢測

本模組提供以下功能：
1. 檢測影像中的手部位置
2. 識別 21 個手部關鍵點（手腕、手指關節、指尖等）
3. 判斷每根手指是伸直還是彎曲
4. 在影像上繪製手部骨架

MediaPipe 手部關鍵點編號說明：
- 點 0: 手腕
- 點 1-4: 大拇指（1:根部, 2:第一關節, 3:第二關節, 4:指尖）
- 點 5-8: 食指（5:根部, 6:第一關節, 7:第二關節, 8:指尖）
- 點 9-12: 中指（9:根部, 10:第一關節, 11:第二關節, 12:指尖）
- 點 13-16: 無名指（13:根部, 14:第一關節, 15:第二關節, 16:指尖）
- 點 17-20: 小指（17:根部, 18:第一關節, 19:第二關節, 20:指尖）
"""
import cv2
import mediapipe as mp
import numpy as np


class HandDetector:
    def __init__(self, 
                 mode=False, 
                 max_hands=1, 
                 detection_confidence=0.7,
                 tracking_confidence=0.5):
        """
        初始化手部檢測器
        
        參數說明:
            mode (bool): 
                - False: 影片模式（默認），適用於連續影像，速度較快
                - True: 靜態圖像模式，每張圖片都重新檢測，準確度較高但速度較慢
            
            max_hands (int): 
                - 最大檢測手數（1-2）
                - 設為 1 可提高性能，適合單手手勢識別
            
            detection_confidence (float): 
                - 檢測信心度閾值（0.0-1.0）
                - 值越高，檢測越嚴格，誤判越少
                - 推薦值：0.7
            
            tracking_confidence (float): 
                - 追蹤信心度閾值（0.0-1.0）
                - 影片模式下，追蹤已檢測手部時使用
                - 值越高，追蹤越穩定
                - 推薦值：0.5
        """
        # 儲存配置參數
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        
        # 初始化 MediaPipe 手部檢測模組
        self.mp_hands = mp.solutions.hands
        
        # 創建手部檢測器實例
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        
        # 用於繪製手部關鍵點和連接線的工具
        self.mp_draw = mp.solutions.drawing_utils
        # 預設的繪製樣式（顏色、線條粗細等）
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
    def find_hands(self, img, draw=True):
        """
        檢測影像中的手部並繪製關鍵點
        
        工作流程：
        1. 將 BGR 影像轉換為 RGB（MediaPipe 需要 RGB 格式）
        2. 使用 MediaPipe 檢測手部
        3. 如果檢測到手部，在影像上繪製 21 個關鍵點和連接線
        
        參數:
            img (numpy.ndarray): 輸入影像 (OpenCV BGR 格式)
            draw (bool): 是否在影像上繪製手部骨架
                        True: 繪製彩色的關鍵點和連接線
                        False: 只檢測不繪製
            
        返回:
            img (numpy.ndarray): 處理後的影像（如果 draw=True 則包含手部骨架）
        """
        # 步驟 1: 轉換顏色空間 BGR -> RGB
        # OpenCV 使用 BGR，但 MediaPipe 需要 RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 步驟 2: 使用 MediaPipe 進行手部檢測
        # self.results 會包含檢測到的所有手部信息
        self.results = self.hands.process(img_rgb)
        
        # 步驟 3: 如果檢測到手部且需要繪製
        if self.results.multi_hand_landmarks:
            # 遍歷所有檢測到的手（通常只有一隻）
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    # 在原始影像上繪製手部關鍵點和連接線
                    self.mp_draw.draw_landmarks(
                        img,                                    # 要繪製的影像
                        hand_landmarks,                         # 手部關鍵點數據
                        self.mp_hands.HAND_CONNECTIONS,        # 關鍵點之間的連接關係
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),  # 關鍵點樣式
                        self.mp_drawing_styles.get_default_hand_connections_style() # 連接線樣式
                    )
        return img
    
    def find_position(self, img, hand_no=0):
        """
        獲取手部關鍵點的像素座標
        
        MediaPipe 返回的是歸一化座標（0.0-1.0），本函數將其轉換為實際像素座標
        
        參數:
            img (numpy.ndarray): 輸入影像，用於獲取尺寸
            hand_no (int): 手部索引
                          0: 第一隻手（默認）
                          1: 第二隻手（如果 max_hands=2）
            
        返回:
            landmark_list (list): 21 個關鍵點的列表
                                格式: [(id, x, y), ...]
                                - id: 關鍵點編號 (0-20)
                                - x: 像素 X 座標
                                - y: 像素 Y 座標
                                
        範例:
            landmark_list = [(0, 320, 240), (1, 305, 235), ...]
            # 點 0 (手腕) 位於 (320, 240)
            # 點 1 (大拇指根部) 位於 (305, 235)
        """
        landmark_list = []
        
        # 檢查是否有檢測到手部
        if self.results.multi_hand_landmarks:
            # 檢查指定的手部索引是否存在
            if hand_no < len(self.results.multi_hand_landmarks):
                # 獲取指定的手部數據
                hand = self.results.multi_hand_landmarks[hand_no]
                
                # 獲取影像尺寸（高度、寬度、通道數）
                h, w, c = img.shape
                
                # 遍歷所有 21 個關鍵點
                for id, landmark in enumerate(hand.landmark):
                    # MediaPipe 返回的是歸一化座標（0.0-1.0）
                    # 轉換為實際像素座標
                    cx = int(landmark.x * w)  # X 座標（寬度方向）
                    cy = int(landmark.y * h)  # Y 座標（高度方向）
                    
                    # 添加到列表：(關鍵點ID, X座標, Y座標)
                    landmark_list.append((id, cx, cy))
                    
        return landmark_list
    
    def fingers_up(self, landmark_list):
        """
        判斷每根手指是伸直還是彎曲（改進版）
        
        判斷原理：
        - 大拇指：比較指尖和根部的水平距離，並區分左右手
        - 其他手指：比較指尖和關節的垂直位置，使用相對距離判斷
        
        參數:
            landmark_list (list): 手部關鍵點列表 [(id, x, y), ...]
                                需要包含所有 21 個關鍵點
            
        返回:
            fingers (list): 5個元素的列表，表示每根手指的狀態
                          格式: [大拇指, 食指, 中指, 無名指, 小指]
                          1: 手指伸直
                          0: 手指彎曲
                          
        範例:
            [0, 1, 1, 0, 0]  # 表示食指和中指伸直（比出數字 2）
            [1, 1, 1, 1, 1]  # 表示所有手指伸直（比出數字 5）
            [0, 0, 0, 0, 0]  # 表示所有手指彎曲（握拳，數字 0）
        """
        # 如果沒有檢測到手部，返回空列表
        if len(landmark_list) == 0:
            return []
        
        fingers = []
        
        # ===== 手指關鍵點 ID 參考 =====
        # 大拇指: 點 1(根), 2, 3, 4(尖)
        # 食指:   點 5(根), 6, 7, 8(尖)
        # 中指:   點 9(根), 10, 11, 12(尖)
        # 無名指: 點 13(根), 14, 15, 16(尖)
        # 小指:   點 17(根), 18, 19, 20(尖)
        
        # 五根手指的指尖 ID
        tip_ids = [4, 8, 12, 16, 20]
        
        # ===== 判斷大拇指（特殊處理）=====
        # 大拇指的方向與其他手指不同（橫向而非縱向）
        # 需要區分左手和右手（左右手大拇指方向相反）
        
        # 獲取手腕（點 0）和中指根部（點 9）的 X 座標，用於判斷左右手
        wrist_x = landmark_list[0][1]
        middle_finger_base_x = landmark_list[9][1]
        
        # 獲取大拇指關鍵點的座標
        thumb_tip_x = landmark_list[4][1]       # 大拇指尖 X 座標
        thumb_tip_y = landmark_list[4][2]       # 大拇指尖 Y 座標
        thumb_ip_x = landmark_list[3][1]        # 大拇指第二關節 X 座標
        thumb_mcp_x = landmark_list[2][1]       # 大拇指第一關節 X 座標
        
        # 判斷是左手還是右手
        # 如果手腕在中指根部的右側，則是左手；否則是右手
        is_left_hand = wrist_x > middle_finger_base_x
        
        # 計算大拇指的水平伸展距離（使用相對距離）
        # 計算手掌寬度作為參考
        palm_width = abs(landmark_list[5][1] - landmark_list[17][1])
        
        # 大拇指伸展判斷（改進版）
        # 方法1：比較指尖和第一關節的水平距離
        thumb_distance = abs(thumb_tip_x - thumb_mcp_x)
        
        # 使用相對閾值（手掌寬度的 15%）而不是固定像素值
        relative_threshold = max(palm_width * 0.15, 20)  # 最小 20 像素
        
        # 方法2：根據左右手判斷大拇指方向
        if is_left_hand:
            # 左手：大拇指尖應該在第一關節的左側
            thumb_extended = (thumb_tip_x < thumb_mcp_x) and (thumb_distance > relative_threshold)
        else:
            # 右手：大拇指尖應該在第一關節的右側
            thumb_extended = (thumb_tip_x > thumb_mcp_x) and (thumb_distance > relative_threshold)
        
        # 額外檢查：大拇指尖也應該遠離手掌中心（點 0）
        thumb_to_wrist_dist = abs(thumb_tip_x - wrist_x)
        thumb_joint_to_wrist_dist = abs(thumb_mcp_x - wrist_x)
        
        # 如果指尖距離手腕更遠，也認為是伸直
        if thumb_to_wrist_dist > thumb_joint_to_wrist_dist * 1.2:
            thumb_extended = True
        
        fingers.append(1 if thumb_extended else 0)
        
        # ===== 判斷其他四根手指（嚴格版 - 檢查所有指節）=====
        # 對於食指、中指、無名指、小指
        # 手指關鍵點結構：
        # - tip_ids[id]: 指尖 (TIP)
        # - tip_ids[id] - 1: 遠端指間關節 (DIP)  
        # - tip_ids[id] - 2: 近端指間關節 (PIP)
        # - tip_ids[id] - 3: 掌指關節根部 (MCP)
        
        for id in range(1, 5):
            # 獲取該手指所有關鍵點的 Y 座標
            tip_id = tip_ids[id]
            tip_y = landmark_list[tip_id][2]        # 指尖
            dip_y = landmark_list[tip_id - 1][2]    # 遠端指間關節（第二關節）
            pip_y = landmark_list[tip_id - 2][2]    # 近端指間關節（第一關節）
            mcp_y = landmark_list[tip_id - 3][2]    # 掌指關節（根部）
            
            # ===== 嚴格的手指伸直判斷 =====
            # 要求：所有指節都必須在正確位置，確保整根手指伸直
            
            # 條件 1：指尖必須在遠端指間關節(DIP)上方
            condition1 = tip_y < dip_y
            
            # 條件 2：遠端指間關節(DIP)必須在近端指間關節(PIP)上方或接近
            # 允許一定誤差（5像素），因為完全伸直時這兩個點可能很接近
            condition2 = dip_y <= pip_y + 5
            
            # 條件 3：近端指間關節(PIP)必須在根部(MCP)上方或接近
            # 這確保手指從根部開始就是伸直的
            condition3 = pip_y <= mcp_y + 10
            
            # 條件 4：指尖必須明顯高於根部（避免手指完全水平或向下）
            # 使用相對距離判斷
            condition4 = tip_y < mcp_y - 10  # 至少高於根部 10 像素
            
            # 額外檢查：計算手指的總長度，確保手指是伸展的
            # 如果手指彎曲，即使各個關節位置正確，總長度也會變短
            total_finger_height = mcp_y - tip_y  # 從根部到指尖的垂直距離
            
            # 獲取手指的理論最小長度（根部到第一關節的距離）
            joint_segment = mcp_y - pip_y
            
            # 如果總高度小於單個關節段的長度，說明手指嚴重彎曲
            condition5 = total_finger_height > joint_segment * 1.5
            
            # ===== 最終判斷 =====
            # 所有條件都必須滿足，才認為手指完全伸直
            if condition1 and condition2 and condition3 and condition4 and condition5:
                fingers.append(1)  # 手指完全伸直
            else:
                fingers.append(0)  # 手指彎曲或部分彎曲
                
        return fingers

