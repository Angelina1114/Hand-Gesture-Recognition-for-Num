"""
手部檢測模組
使用 MediaPipe 進行手部關鍵點檢測
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
        
        參數:
            mode: 靜態圖像模式 (False 為影片模式)
            max_hands: 最大檢測手數
            detection_confidence: 檢測信心度閾值
            tracking_confidence: 追蹤信心度閾值
        """
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        
        # 初始化 MediaPipe 手部模組
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
    def find_hands(self, img, draw=True):
        """
        檢測影像中的手部
        
        參數:
            img: 輸入影像 (BGR 格式)
            draw: 是否繪製手部關鍵點
            
        返回:
            img: 處理後的影像
        """
        # 轉換顏色空間 BGR -> RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        
        # 如果檢測到手部且需要繪製
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        img,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
        return img
    
    def find_position(self, img, hand_no=0):
        """
        獲取手部關鍵點位置
        
        參數:
            img: 輸入影像
            hand_no: 手部索引 (0 為第一隻手)
            
        返回:
            landmark_list: 關鍵點列表 [(id, x, y), ...]
        """
        landmark_list = []
        
        if self.results.multi_hand_landmarks:
            if hand_no < len(self.results.multi_hand_landmarks):
                hand = self.results.multi_hand_landmarks[hand_no]
                
                h, w, c = img.shape
                for id, landmark in enumerate(hand.landmark):
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    landmark_list.append((id, cx, cy))
                    
        return landmark_list
    
    def fingers_up(self, landmark_list):
        """
        判斷哪些手指是伸直的
        
        參數:
            landmark_list: 手部關鍵點列表
            
        返回:
            fingers: 5個元素的列表，1表示手指伸直，0表示彎曲
                    [大拇指, 食指, 中指, 無名指, 小指]
        """
        if len(landmark_list) == 0:
            return []
        
        fingers = []
        
        # 手指關鍵點 ID
        # 大拇指: 1, 2, 3, 4
        # 食指: 5, 6, 7, 8
        # 中指: 9, 10, 11, 12
        # 無名指: 13, 14, 15, 16
        # 小指: 17, 18, 19, 20
        tip_ids = [4, 8, 12, 16, 20]  # 指尖的關鍵點 ID
        
        # 大拇指 (特殊處理，因為方向不同)
        # 使用大拇指指尖(4)和大拇指第二關節(3)的距離來判斷
        thumb_tip_x = landmark_list[tip_ids[0]][1]
        thumb_tip_y = landmark_list[tip_ids[0]][2]
        thumb_joint_x = landmark_list[tip_ids[0] - 1][1]
        thumb_joint_y = landmark_list[tip_ids[0] - 1][2]
        
        # 計算歐幾里得距離
        import math
        thumb_distance = math.sqrt((thumb_tip_x - thumb_joint_x)**2 + (thumb_tip_y - thumb_joint_y)**2)
        
        # 只有當距離足夠大時才認為大拇指伸直（閾值設為50像素）
        if thumb_distance > 50:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # 其他四根手指
        for id in range(1, 5):
            if landmark_list[tip_ids[id]][2] < landmark_list[tip_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
                
        return fingers

