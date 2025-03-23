# myapp/tasks.py

import threading
from .dqn_net.dqn_store import run_dqn_training

TRAINING_STATUS = {
    "running": False,
    "current_episode": 0,
    "total_episodes": 0,
    "avg_buy_price": [],
    "avg_sell_price": [],
    "reward": 0.0,
    "message": ""
}

def _progress_callback(episode, total_episodes, avg_buy_price, avg_sell_price, reward):
    print(f"[Callback] Episode {episode}/{total_episodes}")
    TRAINING_STATUS["current_episode"] = episode
    TRAINING_STATUS["total_episodes"] = total_episodes
    TRAINING_STATUS["avg_buy_price"] = avg_buy_price
    TRAINING_STATUS["avg_sell_price"] = avg_sell_price
    TRAINING_STATUS["reward"] = reward
    TRAINING_STATUS["message"] = f"Episode {episode} / {total_episodes}"
    print(f"🔴 [DEBUG] 更新 TRAINING_STATUS: {TRAINING_STATUS}")

def _background_training(num_episodes, max_steps, product_count):
    TRAINING_STATUS["running"] = True
    TRAINING_STATUS["message"] = "訓練開始"
    TRAINING_STATUS["current_episode"] = 0
    TRAINING_STATUS["avg_buy_price"] = []
    TRAINING_STATUS["avg_sell_price"] = []
    TRAINING_STATUS["reward"] = 0.0

    run_dqn_training(
        num_episodes=num_episodes,
        max_steps=max_steps,
        batch_size=256,
        num_products=product_count,
        progress_callback=_progress_callback
    )

    TRAINING_STATUS["running"] = False
    TRAINING_STATUS["message"] = "訓練完成！"

def start_background_training(num_episodes, max_steps, product_count):
    t = threading.Thread(
        target=_background_training,
        args=(num_episodes, max_steps, product_count)
    )
    t.start()