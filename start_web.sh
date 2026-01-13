#!/bin/bash
# 手勢辨識網頁版快速啟動腳本

echo "======================================"
echo "  手勢數字辨識 Web 系統"
echo "======================================"
echo ""

# 獲取 IP 地址
IP_ADDRESS=$(hostname -I | awk '{print $1}')

echo "正在啟動伺服器..."
echo ""
echo "✅ 請在瀏覽器中訪問："
echo "   本機: http://localhost:5000"
echo "   遠端: http://$IP_ADDRESS:5000"
echo ""
echo "💡 提示："
echo "   - 將手放在攝像頭前方"
echo "   - 保持手勢穩定約 0.5 秒"
echo "   - 按 Ctrl+C 停止伺服器"
echo ""
echo "======================================"
echo ""

# 切換到專案目錄
cd /home/alina/Desktop/數字辨識

# 啟動 Flask 應用
python3 web_app.py

