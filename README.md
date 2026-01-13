# 手勢數字辨識系統

這是一個在 NVIDIA Jetson Orin Nano 上運行的實時手勢數字辨識系統，使用 Logitech C270 攝像頭進行手勢捕捉，可以辨識 0-5 的數字手勢。

## 功能特點

- 🤚 實時手部檢測與追蹤
- 🔢 辨識 0-5 的數字手勢
- 📹 支援 USB 攝像頭（已測試 Logitech C270）
- ⚡ 優化的性能，適合 Jetson Orin Nano
- 🎯 穩定性過濾，減少誤判
- 🖥️ 實時 FPS 顯示

## 系統需求

### 硬件
- NVIDIA Jetson Orin Nano
- USB 攝像頭（例如 Logitech C270）
- 至少 4GB RAM

### 軟件
- Python 3.8+
- OpenCV 4.x
- MediaPipe 0.10.x
- NumPy

## 安裝步驟

### 1. 確認攝像頭連接

```bash
# 檢查攝像頭設備
ls /dev/video*

# 測試攝像頭（可選）
v4l2-ctl --list-devices
```

### 2. 安裝系統依賴

```bash
# 更新套件列表
sudo apt-get update

# 安裝 OpenCV 依賴
sudo apt-get install -y python3-opencv libopencv-dev

# 安裝其他必要工具
sudo apt-get install -y python3-pip python3-dev
```

### 3. 安裝 Python 套件

```bash
# 進入專案目錄
cd /home/alina/Desktop/數字辨識

# 安裝依賴套件
pip3 install -r requirements.txt
```

**注意**: 在 Jetson 設備上，某些套件可能需要從源碼編譯或使用特定版本。如果遇到問題，請參考以下替代方案：

```bash
# 如果 MediaPipe 安裝失敗，可以嘗試：
pip3 install mediapipe --no-deps
pip3 install opencv-contrib-python
```

## 使用方法

### 方式一：桌面版（需要顯示器）

```bash
python3 main.py
```

### 方式二：網頁版（推薦 SSH 使用）

```bash
python3 web_app.py
```

然後在瀏覽器中訪問：
- **本機**: http://localhost:5000
- **遠端**: http://<Jetson的IP地址>:5000

**優點**：
- 可以透過 SSH 使用
- 在任何裝置的瀏覽器中查看
- 美觀的現代化介面
- 實時顯示辨識結果

### 手勢說明

| 數字 | 手勢描述 |
|------|----------|
| 0 | 握拳（所有手指彎曲） |
| 1 | 伸出食指或大拇指 |
| 2 | 伸出食指和中指（剪刀手✌️） |
| 3 | 伸出三根手指 |
| 4 | 伸出四根手指 |
| 5 | 張開手掌（所有手指伸直） |

### 操作說明

- **按 'q' 或 'ESC'**: 退出程式
- **按 'h'**: 在終端顯示幫助信息
- **提示**: 將手放在攝像頭前，保持手勢穩定約 0.5 秒即可識別

## 專案結構

```
數字辨識/
├── main.py                  # 桌面版主程式
├── web_app.py               # 網頁版應用程式
├── hand_detector.py         # 手部檢測模組（基於 MediaPipe）
├── gesture_recognizer.py    # 手勢辨識模組
├── templates/               # 網頁模板
│   └── index.html          # 主頁面
├── static/                  # 靜態資源
├── requirements.txt         # Python 依賴套件
└── README.md               # 說明文檔
```

## 模組說明

### `hand_detector.py`
- **HandDetector 類**: 使用 MediaPipe 進行手部關鍵點檢測
- **主要方法**:
  - `find_hands()`: 檢測並繪製手部關鍵點
  - `find_position()`: 獲取 21 個手部關鍵點的座標
  - `fingers_up()`: 判斷哪些手指是伸直的

### `gesture_recognizer.py`
- **GestureRecognizer 類**: 根據手指狀態識別數字手勢
- **主要方法**:
  - `recognize_number()`: 將手指狀態轉換為數字
  - `get_gesture_description()`: 獲取手勢描述

### `main.py`
- 主程式邏輯，整合攝像頭捕捉、手部檢測和手勢辨識
- 實現穩定性過濾機制
- 提供實時視覺化界面

## 效能優化建議

### 針對 Jetson Orin Nano

1. **降低攝像頭解析度**（如果需要更高 FPS）:
   ```python
   camera_width = 640  # 改為 320
   camera_height = 480  # 改為 240
   ```

2. **調整檢測信心度**:
   ```python
   detector = HandDetector(max_hands=1, detection_confidence=0.6)
   ```

3. **啟用 CUDA 加速**（如果 OpenCV 支援）:
   ```python
   cv2.cuda.setDevice(0)
   ```

## 常見問題

### 問題 1: 攝像頭無法打開
```bash
# 檢查權限
sudo chmod 666 /dev/video0

# 檢查是否被其他程序占用
sudo lsof /dev/video0
```

### 問題 2: FPS 過低
- 降低攝像頭解析度
- 減少檢測的手部數量（設為 1）
- 降低檢測信心度閾值

### 問題 3: 手勢識別不準確
- 確保光線充足
- 保持手部在攝像頭中央
- 手勢保持穩定 0.5 秒以上
- 調整 `stable_threshold` 參數

### 問題 4: MediaPipe 安裝失敗
在 Jetson 上，MediaPipe 可能需要特殊處理：
```bash
# 嘗試安裝預編譯版本
pip3 install mediapipe-0.10.9-cp38-cp38-linux_aarch64.whl
```

## 技術細節

### 手部關鍵點
MediaPipe 檢測 21 個手部關鍵點：
- 點 0: 手腕
- 點 4, 8, 12, 16, 20: 五根手指的指尖
- 其他點: 手指關節

### 手指狀態判斷
- **大拇指**: 比較指尖（點 4）和關節（點 3）的 X 座標
- **其他手指**: 比較指尖和關節的 Y 座標

## 授權

本專案僅供學習和研究使用。

## 貢獻

歡迎提出問題和改進建議！

## 作者

為 NVIDIA Jetson Orin Nano 開發

## 版本歷史

- **v1.0.0** (2026-01-13): 初始版本
  - 基礎手勢辨識功能（0-5）
  - Logitech C270 攝像頭支援
  - 穩定性過濾機制

