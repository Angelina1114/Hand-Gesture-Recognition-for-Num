"""
手勢數字辨識模組
根據手指的伸直/彎曲狀態識別 0-5 的數字手勢

支援的手勢：
- 0: 握拳（所有手指彎曲）
- 1: 伸出一根手指（食指或大拇指）
- 2: 剪刀手（食指+中指，或大拇指+食指）
- 3: 伸出三根手指（多種組合）
- 4: 伸出四根手指
- 5: 張開手掌（所有手指伸直）

演算法原理：
1. 接收 HandDetector 提供的手指狀態（5個0或1的陣列）
2. 統計伸直的手指數量
3. 根據數量和組合判斷具體數字
4. 返回數字和對應的中文名稱
"""


class GestureRecognizer:
    def __init__(self):
        """
        初始化手勢辨識器
        
        建立數字到中文名稱的映射表
        """
        # 數字對應的中文名稱
        self.gesture_names = {
            0: "零",
            1: "一",
            2: "二",
            3: "三",
            4: "四",
            5: "五",
            6: "讚"  # 新增：豎大拇指手勢
        }
    
    def recognize_number(self, fingers):
        """
        根據手指的伸直/彎曲狀態識別數字（0-5）和特殊手勢
        
        嚴格規則：
        - 0: 握拳（所有手指彎曲）
        - 1: 只有食指伸直
        - 2: 食指+中指伸直
        - 3: 食指+中指+無名指 或 中指+無名指+小指
        - 4: 食指+中指+無名指+小指
        - 5: 所有手指伸直
        - 讚: 只有大拇指伸直
        
        參數:
            fingers (list): 5個元素的列表，表示每根手指的狀態
                          格式: [大拇指, 食指, 中指, 無名指, 小指]
                          1: 該手指伸直
                          0: 該手指彎曲
                          
                          範例:
                          [0, 1, 0, 0, 0] → 只有食指 → 數字 1
                          [0, 1, 1, 0, 0] → 食指和中指 → 數字 2
                          [1, 0, 0, 0, 0] → 只有大拇指 → 讚
        
        返回:
            number (int): 識別出的數字或手勢代號
                         0-5: 數字
                         6: 讚
                         -1: 無法識別的手勢
                         
            gesture_name (str): 手勢的中文名稱
        """
        # 檢查輸入是否有效
        if len(fingers) != 5:
            return -1, "未知"
        
        # 為了方便，將 fingers 解構為各個手指的狀態
        thumb, index, middle, ring, pinky = fingers
        
        # 統計伸直的手指總數
        total_fingers_up = sum(fingers)
        
        # ===== 使用嚴格規則進行識別 =====
        
        if total_fingers_up == 0:
            # ===== 數字 0：握拳 =====
            # 所有手指都彎曲 [0, 0, 0, 0, 0]
            return 0, self.gesture_names[0]
            
        elif total_fingers_up == 1:
            # 只有一根手指伸直，需要判斷是哪一根
            
            if thumb == 1 and index == 0:
                # ===== 讚：只有大拇指 =====
                # [1, 0, 0, 0, 0]
                return 6, self.gesture_names[6]
            
            elif index == 1 and thumb == 0:
                # ===== 數字 1：只有食指 =====
                # [0, 1, 0, 0, 0]
                return 1, self.gesture_names[1]
            
            else:
                # 其他單指不符合規則
                return -1, "未知"
                
        elif total_fingers_up == 2:
            # ===== 數字 2：食指+中指 =====
            # 嚴格要求：必須是食指和中指，其他手指都彎曲
            if thumb == 0 and index == 1 and middle == 1 and ring == 0 and pinky == 0:
                # [0, 1, 1, 0, 0]
                return 2, self.gesture_names[2]
            else:
                # 其他兩指組合不符合規則
                return -1, "未知"
                
        elif total_fingers_up == 3:
            # ===== 數字 3：兩種允許的組合 =====
            
            if thumb == 0 and index == 1 and middle == 1 and ring == 1 and pinky == 0:
                # 組合 1: 食指+中指+無名指
                # [0, 1, 1, 1, 0]
                return 3, self.gesture_names[3]
            
            elif thumb == 0 and index == 0 and middle == 1 and ring == 1 and pinky == 1:
                # 組合 2: 中指+無名指+小指
                # [0, 0, 1, 1, 1]
                return 3, self.gesture_names[3]
            
            else:
                # 其他三指組合不符合規則
                return -1, "未知"
                
        elif total_fingers_up == 4:
            # ===== 數字 4：食指+中指+無名指+小指 =====
            # 嚴格要求：大拇指彎曲，其他四指伸直
            if thumb == 0 and index == 1 and middle == 1 and ring == 1 and pinky == 1:
                # [0, 1, 1, 1, 1]
                return 4, self.gesture_names[4]
            else:
                # 其他四指組合不符合規則
                return -1, "未知"
            
        elif total_fingers_up == 5:
            # ===== 數字 5：張開手掌 =====
            # 所有手指都伸直 [1, 1, 1, 1, 1]
            return 5, self.gesture_names[5]
            
        else:
            # 不應該出現的情況
            return -1, "未知"
    
    def get_gesture_description(self, number):
        """
        獲取數字手勢的詳細描述
        
        用於顯示在網頁或終端上，幫助用戶了解如何比出每個數字
        
        參數:
            number (int): 數字 (0-5) 或特殊手勢代號 (6)
            
        返回:
            description (str): 該數字的手勢描述
            
        範例:
            get_gesture_description(2) → "食指+中指"
        """
        # 每個數字對應的詳細描述（嚴格規則）
        descriptions = {
            0: "握拳（所有手指彎曲）",
            1: "只伸出食指",
            2: "食指+中指",
            3: "食指+中指+無名指 或 中指+無名指+小指",
            4: "食指+中指+無名指+小指",
            5: "張開手掌（所有手指伸直）",
            6: "只伸出大拇指（讚）"
        }
        
        # 返回對應描述，如果數字不在範圍內則返回"未知手勢"
        return descriptions.get(number, "未知手勢")

