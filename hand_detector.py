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
import math


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
    
    def get_hand_count(self):
        """
        獲取檢測到的手部數量
        
        返回:
            count (int): 手部數量 (0, 1, 或 2)
        """
        if self.results.multi_hand_landmarks:
            return len(self.results.multi_hand_landmarks)
        return 0
    
    def vector_2d_angle(self, v1, v2):
        """
        計算兩個二維向量之間的夾角
        
        使用向量夾角公式：
        angle = arccos((v1 · v2) / (|v1| * |v2|))
        
        參數:
            v1: 向量1 (x, y)
            v2: 向量2 (x, y)
            
        返回:
            angle (float): 角度（度數，0-180度）
        """
        v1_x, v1_y = v1[0], v1[1]
        v2_x, v2_y = v2[0], v2[1]
        
        try:
            # 計算向量點積
            dot_product = v1_x * v2_x + v1_y * v2_y
            # 計算向量長度
            v1_length = math.sqrt(v1_x ** 2 + v1_y ** 2)
            v2_length = math.sqrt(v2_x ** 2 + v2_y ** 2)
            # 計算夾角
            cos_angle = dot_product / (v1_length * v2_length)
            # 限制在 [-1, 1] 範圍內
            cos_angle = max(-1, min(1, cos_angle))
            # 轉換為度數
            angle = math.degrees(math.acos(cos_angle))
        except:
            angle = 180
        
        return angle
    
    def hand_angle(self, landmark_list):
        """
        根據 21 個手部關鍵點，計算每根手指的角度
        
        計算方法：
        - 向量1：從手腕(點0)指向手指的某個關節
        - 向量2：從關節指向指尖
        - 計算這兩個向量的夾角
        
        參數:
            landmark_list (list): 手部關鍵點列表 [(id, x, y), ...]
            
        返回:
            angle_list (list): 5 個手指的角度 [大拇指, 食指, 中指, 無名指, 小指]
        """
        angle_list = []
        
        # 大拇指角度
        angle = self.vector_2d_angle(
            (landmark_list[0][1] - landmark_list[2][1], 
             landmark_list[0][2] - landmark_list[2][2]),
            (landmark_list[3][1] - landmark_list[4][1], 
             landmark_list[3][2] - landmark_list[4][2])
        )
        angle_list.append(angle)
        
        # 食指角度
        angle = self.vector_2d_angle(
            (landmark_list[0][1] - landmark_list[6][1], 
             landmark_list[0][2] - landmark_list[6][2]),
            (landmark_list[7][1] - landmark_list[8][1], 
             landmark_list[7][2] - landmark_list[8][2])
        )
        angle_list.append(angle)
        
        # 中指角度
        angle = self.vector_2d_angle(
            (landmark_list[0][1] - landmark_list[10][1], 
             landmark_list[0][2] - landmark_list[10][2]),
            (landmark_list[11][1] - landmark_list[12][1], 
             landmark_list[11][2] - landmark_list[12][2])
        )
        angle_list.append(angle)
        
        # 無名指角度
        angle = self.vector_2d_angle(
            (landmark_list[0][1] - landmark_list[14][1], 
             landmark_list[0][2] - landmark_list[14][2]),
            (landmark_list[15][1] - landmark_list[16][1], 
             landmark_list[15][2] - landmark_list[16][2])
        )
        angle_list.append(angle)
        
        # 小指角度
        angle = self.vector_2d_angle(
            (landmark_list[0][1] - landmark_list[18][1], 
             landmark_list[0][2] - landmark_list[18][2]),
            (landmark_list[19][1] - landmark_list[20][1], 
             landmark_list[19][2] - landmark_list[20][2])
        )
        angle_list.append(angle)
        
        return angle_list
    
    def fingers_up(self, landmark_list):
        """
        判斷每根手指是伸直還是彎曲（使用向量夾角判斷）
        
        判斷原理（參考優化方法）：
        - 計算從手腕到關節的向量，與關節到指尖的向量之間的夾角
        - 角度 < 50度：手指伸直（向量接近同向）
        - 角度 >= 50度：手指彎曲（向量夾角變大）
        
        參數:
            landmark_list (list): 手部關鍵點列表 [(id, x, y), ...]
                                需要包含所有 21 個關鍵點
            
        返回:
            fingers (list): 5個元素的列表，表示每根手指的狀態
                          格式: [大拇指, 食指, 中指, 無名指, 小指]
                          1: 手指伸直（角度 < 50度）
                          0: 手指彎曲（角度 >= 50度）
                          
        範例:
            [0, 1, 1, 0, 0]  # 表示食指和中指伸直（比出數字 2）
            [1, 1, 1, 1, 1]  # 表示所有手指伸直（比出數字 5）
            [0, 0, 0, 0, 0]  # 表示所有手指彎曲（握拳，數字 0）
        """
        # 如果沒有檢測到手部，返回空列表
        if len(landmark_list) == 0:
            return []
        
        # 角度閾值：小於此角度視為伸直，大於等於此角度視為彎曲
        ANGLE_THRESHOLD = 50  # 度
        
        # 計算所有手指的角度
        finger_angles = self.hand_angle(landmark_list)
        
        # 根據角度判斷每根手指是否伸直
        fingers = []
        for angle in finger_angles:
            if angle < ANGLE_THRESHOLD:
                fingers.append(1)  # 手指伸直
            else:
                fingers.append(0)  # 手指彎曲
        
        return fingers

