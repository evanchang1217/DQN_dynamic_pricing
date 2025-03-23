# myapp/views.py

from django.shortcuts import render
from django.http import JsonResponse
import os
from django.conf import settings

from .tasks import TRAINING_STATUS, start_background_training
from .dqn_net import dqn_store

def start_training_view(request):
    # 每次 GET 時先刪除舊圖，避免舊圖被前端顯示
    file_path = os.path.join(settings.BASE_DIR, 'statics', 'images', 'dqn_reward.png')
    if os.path.exists(file_path):
        os.remove(file_path)

    message = None
    product_data = []
    # 預設商品數量
    product_count = 3
    if request.method == "POST":
        try:
            num_episodes = int(request.POST.get("num_episodes",))
            max_steps = int(request.POST.get("max_steps", ))
            product_count = int(request.POST.get("product_count", ))

            # 讀取每個商品的價格上下限
            new_buy_bounds = []
            new_sell_bounds = []
            for i in range(product_count):
                buy_low = float(request.POST.get(f"buy_low_{i}"))
                buy_high = float(request.POST.get(f"buy_high_{i}"))
                sell_low = float(request.POST.get(f"sell_low_{i}"))
                sell_high = float(request.POST.get(f"sell_high_{i}"))
                new_buy_bounds.append((buy_low, buy_high))
                new_sell_bounds.append((sell_low, sell_high))


            # 存入 product_data（用於重新渲染頁面）
            product_data.append({
                "buy_low": buy_low,
                "buy_high": buy_high,
                "sell_low": sell_low,
                "sell_high": sell_high,
            })

            # 轉成 float 後存進 DQN 的全域變數
            new_buy_bounds.append((float(buy_low), float(buy_high)))
            new_sell_bounds.append((float(sell_low), float(sell_high)))

            # 更新 dqn_store 模組的全域變數
            dqn_store.buy_price_bounds = new_buy_bounds
            dqn_store.sell_price_bounds = new_sell_bounds

            # 啟動背景訓練
            start_background_training(num_episodes, max_steps, product_count)
            message = "已開始訓練，請等待..."
        except Exception as e:
            message = f"輸入錯誤：{e}"
    else:
        # GET 請求時，先給預設空值
        for i in range(product_count):
            product_data.append({
                "buy_low": "",
                "buy_high": "",
                "sell_low": "",
                "sell_high": "",
            })

    # 將需要的資料帶回模板
    context = {
        "message": message,
        "product_count": product_count,
        "product_data": product_data,
    }
    return render(request, "training.html", {"message": message})

def training_status_api(request):
    response = JsonResponse(TRAINING_STATUS)
    # 加上不快取標頭，避免前端讀取到舊資料
    response['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response


'''
# myapp/views.py

from django.shortcuts import render
from .dqn_net.dqn_store import run_dqn_training
import os
from django.conf import settings

def dqn_training_view(request):
    # 當收到 GET 請求時，清除之前的訓練圖
    file_path = os.path.join(settings.BASE_DIR, 'statics', 'images', 'dqn_reward.png')
    if os.path.exists(file_path):
        os.remove(file_path)
    
    result = None
    if request.method == "POST":
        try:
            # 讀取商品數量
            product_count = int(request.POST.get("product_count"))
            new_buy_bounds = []
            new_sell_bounds = []
            for i in range(product_count):
                buy_low = float(request.POST.get(f"buy_low_{i}"))
                buy_high = float(request.POST.get(f"buy_high_{i}"))
                sell_low = float(request.POST.get(f"sell_low_{i}"))
                sell_high = float(request.POST.get(f"sell_high_{i}"))
                new_buy_bounds.append((buy_low, buy_high))
                new_sell_bounds.append((sell_low, sell_high))
            # 更新 dqn_store 模組中的全域價格區間（延後導入以避免循環引用）
            from .dqn_net import dqn_store
            dqn_store.buy_price_bounds = new_buy_bounds
            dqn_store.sell_price_bounds = new_sell_bounds

            # 執行 DQN 訓練，同時將商品數量作為參數傳入
            run_dqn_training(num_episodes=300, max_steps=100, num_products=product_count)
            result = "DQN 訓練完成，請查看 Reward 圖。"
        except Exception as e:
            result = f"輸入錯誤：{e}"
    return render(request, "dqn_training.html", {"result": result})

'''




def calculate_view(request):
    result = None  # 初始化結果變數

    if request.method == "POST":
        num1 = request.POST.get("num1")
        num2 = request.POST.get("num2")

        if num1 and num2:  # 確保輸入有效
            try:
                num1 = float(num1)
                num2 = float(num2)
                result = num1 + num2  # 直接計算（不存資料庫）
            except ValueError:
                result = "請輸入有效數字！"

    return render(request, "calculate_view.html", {"result": result})


#多執行續
#輸入當天價格跑出明天價格

'''
def dqn_training_view(request):
    # 當收到 GET 請求時，清除之前的訓練圖與價格資料
    file_path = os.path.join(settings.BASE_DIR, 'statics/images', 'dqn_reward.png')
    if os.path.exists(file_path):
        os.remove(file_path)
    
    # 重置全域價格區間（這裡用預設值重置）
    global buy_price_bounds, sell_price_bounds
    buy_price_bounds = [(100, 150), (110, 140), (105, 160), (95, 145), (115, 155)]
    sell_price_bounds = [(120, 170), (125, 165), (130, 175), (115, 160), (135, 180)]
    
    result = None
    if request.method == "POST":
        try:
            new_buy_bounds = []
            new_sell_bounds = []
            for i in range(5):
                buy_low = float(request.POST.get(f"buy_low_{i}"))
                buy_high = float(request.POST.get(f"buy_high_{i}"))
                sell_low = float(request.POST.get(f"sell_low_{i}"))
                sell_high = float(request.POST.get(f"sell_high_{i}"))
                new_buy_bounds.append((buy_low, buy_high))
                new_sell_bounds.append((sell_low, sell_high))
            # 更新全域價格區間
            buy_price_bounds = new_buy_bounds
            sell_price_bounds = new_sell_bounds

            # 執行 DQN 訓練（同步阻塞方式）
            run_dqn_training(num_episodes=500, max_steps=100)
            result = "DQN 訓練完成，請查看 Reward 圖。"
        except Exception as e:
            result = f"輸入錯誤：{e}"
    return render(request, "dqn_training.html", {"result": result})'''