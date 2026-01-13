"""
手勢數字辨識模組
根據手指狀態識別 0-5 的數字手勢
"""


class GestureRecognizer:
    def __init__(self):
        """初始化手勢辨識器"""
        self.gesture_names = {
            0: "零",
            1: "一",
            2: "二",
            3: "三",
            4: "四",
            5: "五"
        }
    
    def recognize_number(self, fingers):
        """
        根據手指狀態識別數字
        
        參數:
            fingers: 5個元素的列表，1表示手指伸直，0表示彎曲
                    [大拇指, 食指, 中指, 無名指, 小指]
        
        返回:
            number: 識別出的數字 (0-5)，如果無法識別則返回 -1
            gesture_name: 數字的中文名稱
        """
        if len(fingers) != 5:
            return -1, "未知"
        
        # 計算伸直的手指數量
        total_fingers_up = sum(fingers)
        
        # 根據伸直的手指數量判斷數字
        if total_fingers_up == 0:
            # 零 - 所有手指都彎曲（握拳）
            number = 0
        elif total_fingers_up == 1:
            # 一 - 只有食指伸直
            if fingers[1] == 1:
                number = 1
            # 或者只有大拇指伸直（點讚手勢）
            elif fingers[0] == 1:
                number = 1
            else:
                return -1, "未知"
        elif total_fingers_up == 2:
            # 二 - 食指和中指伸直（勝利/和平手勢）
            if fingers[1] == 1 and fingers[2] == 1:
                number = 2
            # 或者大拇指和食指伸直
            elif fingers[0] == 1 and fingers[1] == 1:
                number = 2
            else:
                return -1, "未知"
        elif total_fingers_up == 3:
            # 三 - 食指、中指、無名指伸直
            if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
                number = 3
            # 或者大拇指、食指、中指伸直
            elif fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 1:
                number = 3
            else:
                return -1, "未知"
        elif total_fingers_up == 4:
            # 四 - 四根手指伸直（大拇指彎曲或其他）
            number = 4
        elif total_fingers_up == 5:
            # 五 - 所有手指伸直（張開手掌）
            number = 5
        else:
            return -1, "未知"
        
        gesture_name = self.gesture_names.get(number, "未知")
        return number, gesture_name
    
    def get_gesture_description(self, number):
        """
        獲取手勢的描述信息
        
        參數:
            number: 數字 (0-5)
            
        返回:
            description: 手勢描述
        """
        descriptions = {
            0: "握拳（所有手指彎曲）",
            1: "伸出食指或大拇指",
            2: "伸出食指和中指（剪刀手）",
            3: "伸出三根手指",
            4: "伸出四根手指",
            5: "張開手掌（所有手指伸直）"
        }
        return descriptions.get(number, "未知手勢")

