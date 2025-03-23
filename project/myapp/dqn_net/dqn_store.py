# myapp/dqn_net/dqn_store.py

import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
from collections import deque

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from django.conf import settings

# 全域變數（由外部設定），儲存每個商品的買/賣價上下限
buy_price_bounds = []
sell_price_bounds = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

output_dim = 25
minimum_offset = -0.05
maximum_offset = 0.05
div_dim = int(output_dim ** 0.5)
offset_range_per_step = (maximum_offset - minimum_offset) / (div_dim - 1)
action_offsets = [minimum_offset + offset * offset_range_per_step for offset in range(div_dim)]

class StoreEnv(gym.Env):
    def __init__(self, num_products, num_players=500):
        super(StoreEnv, self).__init__()
        self.num_products = num_products
        self.num_players = num_players
        self.state = None
        self.reset()

    def _calculate_probability(self, price, low, high, is_buying):
        price = np.clip(price, low, high)
        if is_buying:
            return max(0, min(1, (price - low) / (high - low)))
        else:
            return max(0, min(1, 1 - (price - low) / (high - low)))

    def step(self, actions):
        next_state = self.state.copy()
        total_buy_count = []
        total_sell_count = []

        for i in range(self.num_products):
            action = (actions[i] // div_dim, actions[i] % div_dim)
            buy_adj = action_offsets[action[0]]
            sell_adj = action_offsets[action[1]]

            bp_low, bp_high = buy_price_bounds[i]
            sp_low, sp_high = sell_price_bounds[i]

            # 更新買賣價格
            next_state[i, 0] = np.clip(self.state[i, 0] * buy_adj + self.state[i, 0], bp_low, bp_high)
            next_state[i, 1] = np.clip(self.state[i, 1] * sell_adj + self.state[i, 1], sp_low, sp_high)

            buy_prob = self._calculate_probability(next_state[i, 0], bp_low, bp_high, True)
            sell_prob = self._calculate_probability(next_state[i, 1], sp_low, sp_high, False)

            buy_count = sum(np.random.rand() < buy_prob for _ in range(self.num_players))
            sell_count = sum(np.random.rand() < sell_prob for _ in range(self.num_players))

            total_buy_count.append(buy_count)
            total_sell_count.append(sell_count)
            next_state[i, 2] = buy_count
            next_state[i, 3] = sell_count

        # 獎勵函式(範例)
        rewards = [
            -abs(b - self.num_players / 2) - abs(s - self.num_players / 2)
            for b, s in zip(total_buy_count, total_sell_count)
        ]
        done = False
        info = {}
        self.state = next_state.copy()
        return next_state, rewards, done, info

    def reset(self):
        state_list = []
        for i in range(self.num_products):
            bp_low, bp_high = buy_price_bounds[i]
            sp_low, sp_high = sell_price_bounds[i]
            buy = random.randint(bp_low, bp_high)
            sell = random.randint(sp_low, sp_high)
            state_list.append([buy, sell, 0, 0])
        self.state = np.array(state_list)
        return self.state

class DQN(nn.Module):
    def __init__(self, input_dim):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 256)
        self.layer4 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)

def run_dqn_training(num_episodes, max_steps, batch_size, num_products, progress_callback=None):
    env = StoreEnv(num_products=num_products)
    input_dim = 4 + num_products
    policy_net = DQN(input_dim).to(device)
    target_net = DQN(input_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.AdamW(policy_net.parameters(), lr=0.001)

    class ReplayBuffer:
        def __init__(self, capacity):
            self.capacity = capacity
            self.buffer = deque(maxlen=capacity)
        def push(self, state, action, reward, next_state, done):
            self.buffer.append((state, action, reward, next_state, done))
        def sample(self, batch_size):
            return random.sample(self.buffer, batch_size)
        def __len__(self):
            return len(self.buffer)

    buffer = ReplayBuffer(10000)

    gamma = 0.99
    epsilon = 0.99
    epsilon_decay = 0.99
    min_epsilon = 0.1
    target_update = 10

    rewards_history = [[] for _ in range(num_products)]
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))

    def select_action(state, product_idx, epsilon):
        product_one_hot = np.zeros(num_products)
        product_one_hot[product_idx] = 1
        combined_state = np.concatenate((state, product_one_hot))
        if random.random() < epsilon:
            return random.randint(0, output_dim - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(combined_state, dtype=torch.float32).to(device)
                return policy_net(state_tensor).argmax().item()

    def optimize_model():
        if len(buffer) < batch_size:
            return
        transitions = buffer.sample(batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

        batch_state = torch.tensor(np.array(batch_state), dtype=torch.float32).to(device)
        batch_action = torch.tensor(np.array(batch_action), dtype=torch.int64).unsqueeze(1).to(device)
        batch_reward = torch.tensor(np.array(batch_reward), dtype=torch.float32).unsqueeze(1).to(device)
        batch_next_state = torch.tensor(np.array(batch_next_state), dtype=torch.float32).to(device)
        batch_done = torch.tensor(np.array(batch_done), dtype=torch.float32).unsqueeze(1).to(device)

        current_q = policy_net(batch_state).gather(1, batch_action)
        with torch.no_grad():
            next_q = target_net(batch_next_state).max(1)[0].unsqueeze(1)
        expected_q = batch_reward + gamma * next_q * (1 - batch_done)

        loss = F.smooth_l1_loss(current_q, expected_q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("進入 run_dqn_training")

    for episode in range(1, num_episodes + 1):
        print(f"Episode loop: {episode}")
        total_rewards = [0] * num_products
        states = env.reset()

        for t in range(max_steps):
            actions = []
            for i in range(num_products):
                a = select_action(states[i], i, epsilon)
                actions.append(a)

            next_states, rewards, done, _ = env.step(actions)
            for i in range(num_products):
                product_one_hot = np.zeros(num_products)
                product_one_hot[i] = 1
                combined_state = np.concatenate((states[i], product_one_hot))
                combined_next_state = np.concatenate((next_states[i], product_one_hot))
                buffer.push(combined_state, actions[i], rewards[i], combined_next_state, done)
                total_rewards[i] += rewards[i]

            states = next_states

        # episode 結束後進行一次訓練
        optimize_model()

        for i in range(num_products):
            rewards_history[i].append(total_rewards[i])

        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # 更新圖表
        ax.clear()
        for i in range(num_products):
            ax.plot(range(len(rewards_history[i])), rewards_history[i], label=f'Product {i} Reward')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.legend()
        plot_path = os.path.join(settings.BASE_DIR, 'statics', 'images', 'dqn_reward.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.draw()
        plt.pause(0.01)

        # 若有 callback 就回報進度
        if progress_callback:
            avg_buy_prices = [env.state[i, 0] for i in range(num_products)]
            avg_sell_prices = [env.state[i, 1] for i in range(num_products)]
        progress_callback(
            episode=episode,
            total_episodes=num_episodes,
            avg_buy_price=[int(price) for price in avg_buy_prices],  # 轉為純 Python int
            avg_sell_price=[int(price) for price in avg_sell_prices],  # 轉為純 Python int
            reward=int(sum(total_rewards))  # 轉為 int
        )

        # 讓 Python 有機會處理其他任務
        time.sleep(0.05)

    plt.ioff()

'''


import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
from collections import deque
import matplotlib.pyplot as plt

# 定義運算裝置
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# 商品數量與行動空間設定
NUM_PRODUCTS = 5
NUM_PLAYERS = 500

# 全域價格區間，會由 views 更新
buy_price_bounds = [(100, 150), (110, 140), (105, 160), (95, 145), (115, 155)]
sell_price_bounds = [(120, 170), (125, 165), (130, 175), (115, 160), (135, 180)]

output_dim = 25  # 每個商品的行動空間大小
minimum_offset = -0.05
maximum_offset = 0.05
div_dim = int(output_dim ** 0.5)
offset_range_per_step = (maximum_offset - minimum_offset) / (div_dim - 1)
action_offsets = [minimum_offset + offset * offset_range_per_step for offset in range(div_dim)]

# 定義環境
class StoreEnv(gym.Env):
    def __init__(self, num_products=NUM_PRODUCTS, num_players=NUM_PLAYERS):
        super(StoreEnv, self).__init__()
        self.num_products = num_products
        self.num_players = num_players
        self.state = None
        self.reset()

    def _calculate_probability(self, price, low, high, is_buying):
        price = np.clip(price, low, high)
        if is_buying:
            return max(0, min(1, (price - low) / (high - low)))
        else:
            return max(0, min(1, 1 - (price - low) / (high - low)))
        
    def step(self, actions):
        next_state = self.state.copy()
        total_buy_count = []
        total_sell_count = []
        for i in range(self.num_products):
            # 將 action 拆解成兩個編號，分別對應買賣調整
            action = (actions[i] // div_dim, actions[i] % div_dim)
            buy_adj = action_offsets[action[0]]
            sell_adj = action_offsets[action[1]]
            bp_low, bp_high = buy_price_bounds[i]
            sp_low, sp_high = sell_price_bounds[i]
            next_state[i, 0] = np.clip(self.state[i, 0] * buy_adj + self.state[i, 0], bp_low, bp_high)
            next_state[i, 1] = np.clip(self.state[i, 1] * sell_adj + self.state[i, 1], sp_low, sp_high)
            buy_prob = self._calculate_probability(next_state[i, 0], bp_low, bp_high, True)
            sell_prob = self._calculate_probability(next_state[i, 1], sp_low, sp_high, False)
            buy_count = sum(np.random.rand() < buy_prob for _ in range(self.num_players))
            sell_count = sum(np.random.rand() < sell_prob for _ in range(self.num_players))
            total_buy_count.append(buy_count)
            total_sell_count.append(sell_count)
            next_state[i, 2] = buy_count
            next_state[i, 3] = sell_count
        rewards = [ -abs(bc - self.num_players/2) - abs(sc - self.num_players/2)
                    for bc, sc in zip(total_buy_count, total_sell_count)]
        done = False
        info = {}
        self.state = next_state.copy()
        return next_state, rewards, done, info

    def reset(self):
        state_list = []
        for i in range(self.num_products):
            bp_low, bp_high = buy_price_bounds[i]
            sp_low, sp_high = sell_price_bounds[i]
            buy = random.randint(bp_low, bp_high)
            sell = random.randint(sp_low, sp_high)
            state_list.append([buy, sell, 0, 0])
        self.state = np.array(state_list)
        return self.state

# DQN 模型：輸入層結合原始狀態 (4 維) 與 one-hot 編碼 (NUM_PRODUCTS 維)
class DQN(nn.Module):
    def __init__(self, input_dim):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 256)
        self.layer4 = nn.Linear(256, output_dim)
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        return self.layer4(x)

input_dim = 4 + NUM_PRODUCTS  # 4 維狀態 + 5 維 one-hot 編碼

# ReplayBuffer 定義
class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)

# 以下是 DQN 訓練流程
def run_dqn_training(num_episodes, max_steps, batch_size=256):
    env = StoreEnv(num_products=NUM_PRODUCTS)
    policy_net = DQN(input_dim).to(device)
    target_net = DQN(input_dim).to(device)
    optimizer = optim.AdamW(policy_net.parameters(), lr=0.001)
    buffer = ReplayBuffer(10000)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    gamma = 0.99
    epsilon = 0.99
    epsilon_decay = 0.996
    min_epsilon = 0.1
    target_update = 10

    rewards_history = [[] for _ in range(NUM_PRODUCTS)]
    plt.ion()
    fig, ax = plt.subplots()

    # 定義 action 選擇函式：結合狀態與 one-hot 編碼
    def select_action(state, product_idx, epsilon):
        product_one_hot = np.zeros(NUM_PRODUCTS)
        product_one_hot[product_idx] = 1
        combined_state = np.concatenate((state, product_one_hot))
        if random.random() < epsilon:
            return int(random.random() * output_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(combined_state, dtype=torch.float32).to(device)
                return policy_net(state_tensor).argmax().item()

    # 定義模型更新函式
    def optimize_model():
        if len(buffer) < batch_size:
            return
        transitions = buffer.sample(batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)
        batch_state = torch.tensor(np.array(batch_state), dtype=torch.float32).to(device)
        batch_action = torch.tensor(np.array(batch_action), dtype=torch.int64).unsqueeze(1).to(device)
        batch_reward = torch.tensor(np.array(batch_reward), dtype=torch.float32).unsqueeze(1).to(device)
        batch_next_state = torch.tensor(np.array(batch_next_state), dtype=torch.float32).to(device)
        batch_done = torch.tensor(np.array(batch_done), dtype=torch.float32).unsqueeze(1).to(device)
        current_q = policy_net(batch_state).gather(1, batch_action)
        with torch.no_grad():
            next_q = target_net(batch_next_state).max(1)[0].unsqueeze(1)
        expected_q = batch_reward + gamma * next_q * (1 - batch_done)
        loss = F.smooth_l1_loss(current_q, expected_q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for episode in range(num_episodes):
        total_rewards = [0] * NUM_PRODUCTS
        states = env.reset()  # shape: (NUM_PRODUCTS, 4)
        for t in range(max_steps):
            actions = []
            for i in range(NUM_PRODUCTS):
                a = select_action(states[i], i, epsilon)
                actions.append(a)
            next_states, rewards, done, _ = env.step(actions)
            for i in range(NUM_PRODUCTS):
                # 結合狀態與 one-hot 編碼後存入 buffer
                product_one_hot = np.zeros(NUM_PRODUCTS)
                product_one_hot[i] = 1
                combined_state = np.concatenate((states[i], product_one_hot))
                combined_next_state = np.concatenate((next_states[i], product_one_hot))
                buffer.push(combined_state, actions[i], rewards[i], combined_next_state, done)
                total_rewards[i] += rewards[i]
            states = next_states

        for i in range(NUM_PRODUCTS):
            rewards_history[i].append(total_rewards[i])

        optimize_model()

        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # 更新 Reward 圖，每個 episode 後存檔至 static/dqn_reward.png
        ax.clear()
        for i in range(NUM_PRODUCTS):
            ax.plot(range(len(rewards_history[i])), rewards_history[i], label=f'Product {i} Reward')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.legend()
        plot_path = os.path.join('statics/images', 'dqn_reward.png')
        plt.savefig(plot_path)
        plt.draw()
        plt.pause(0.01)

    plt.ioff()
    plt.show()
    # 儲存訓練後模型
    torch.save(policy_net.state_dict(), 'dqn_policy_net.pth')
'''


'''

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
from collections import deque
import matplotlib.pyplot as plt
from django.conf import settings

# 定義運算裝置
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# 商品數量與行動空間設定
NUM_PLAYERS = 500

# 全域價格區間，會由 views 更新，不設定預設值
buy_price_bounds = []
sell_price_bounds = []

output_dim = 25  # 每個商品的行動空間大小
minimum_offset = -0.05
maximum_offset = 0.05
div_dim = int(output_dim ** 0.5)
offset_range_per_step = (maximum_offset - minimum_offset) / (div_dim - 1)
action_offsets = [minimum_offset + offset * offset_range_per_step for offset in range(div_dim)]

# 定義環境，接受 num_products 參數
class StoreEnv(gym.Env):
    def __init__(self, num_products, num_players=NUM_PLAYERS):
        super(StoreEnv, self).__init__()
        self.num_products = num_products
        self.num_players = num_players
        self.state = None
        self.reset()

    def _calculate_probability(self, price, low, high, is_buying):
        price = np.clip(price, low, high)
        if is_buying:
            return max(0, min(1, (price - low) / (high - low)))
        else:
            return max(0, min(1, 1 - (price - low) / (high - low)))
        
    def step(self, actions):
        next_state = self.state.copy()
        total_buy_count = []
        total_sell_count = []
        for i in range(self.num_products):
            action = (actions[i] // div_dim, actions[i] % div_dim)
            buy_adj = action_offsets[action[0]]
            sell_adj = action_offsets[action[1]]
            # 這裡假設 buy_price_bounds 與 sell_price_bounds 已經由前端更新，且列表長度正確
            bp_low, bp_high = buy_price_bounds[i]
            sp_low, sp_high = sell_price_bounds[i]
            next_state[i, 0] = np.clip(self.state[i, 0] * buy_adj + self.state[i, 0], bp_low, bp_high)
            next_state[i, 1] = np.clip(self.state[i, 1] * sell_adj + self.state[i, 1], sp_low, sp_high)
            buy_prob = self._calculate_probability(next_state[i, 0], bp_low, bp_high, True)
            sell_prob = self._calculate_probability(next_state[i, 1], sp_low, sp_high, False)
            buy_count = sum(np.random.rand() < buy_prob for _ in range(self.num_players))
            sell_count = sum(np.random.rand() < sell_prob for _ in range(self.num_players))
            total_buy_count.append(buy_count)
            total_sell_count.append(sell_count)
            next_state[i, 2] = buy_count
            next_state[i, 3] = sell_count
        rewards = [ -abs(bc - self.num_players/2) - abs(sc - self.num_players/2)
                    for bc, sc in zip(total_buy_count, total_sell_count)]
        done = False
        info = {}
        self.state = next_state.copy()
        return next_state, rewards, done, info

    def reset(self):
        state_list = []
        for i in range(self.num_products):
            bp_low, bp_high = buy_price_bounds[i]
            sp_low, sp_high = sell_price_bounds[i]
            buy = random.randint(bp_low, bp_high)
            sell = random.randint(sp_low, sp_high)
            state_list.append([buy, sell, 0, 0])
        self.state = np.array(state_list)
        return self.state

# DQN 模型：輸入層結合原始狀態 (4 維) 與商品 one-hot 編碼 (num_products 維)
class DQN(nn.Module):
    def __init__(self, input_dim):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 256)
        self.layer4 = nn.Linear(256, output_dim)
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        return self.layer4(x)

# run_dqn_training 現在接受 num_products 作為參數
def run_dqn_training(num_episodes, max_steps, batch_size=256, num_products=5):
    env = StoreEnv(num_products=num_products)
    # 更新模型輸入維度：原始狀態 4 維 + num_products (one-hot)
    input_dim = 4 + num_products
    policy_net = DQN(input_dim).to(device)
    target_net = DQN(input_dim).to(device)
    optimizer = optim.AdamW(policy_net.parameters(), lr=0.001)
    
    # 自定義 ReplayBuffer
    class ReplayBuffer(object):
        def __init__(self, capacity):
            self.capacity = capacity
            self.buffer = deque(maxlen=capacity)
        def push(self, state, action, reward, next_state, done):
            self.buffer.append((state, action, reward, next_state, done))
        def sample(self, batch_size):
            return random.sample(self.buffer, batch_size)
        def __len__(self):
            return len(self.buffer)
    buffer = ReplayBuffer(10000)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    gamma = 0.99
    epsilon = 0.99
    epsilon_decay = 0.996
    min_epsilon = 0.1
    target_update = 10

    rewards_history = [[] for _ in range(num_products)]
    plt.ion()
    fig, ax = plt.subplots()

    # 定義 action 選擇函式
    def select_action(state, product_idx, epsilon):
        product_one_hot = np.zeros(num_products)
        product_one_hot[product_idx] = 1
        combined_state = np.concatenate((state, product_one_hot))
        if random.random() < epsilon:
            return int(random.random() * output_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(combined_state, dtype=torch.float32).to(device)
                return policy_net(state_tensor).argmax().item()

    # 定義模型更新函式
    def optimize_model():
        if len(buffer) < batch_size:
            return
        transitions = buffer.sample(batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)
        batch_state = torch.tensor(np.array(batch_state), dtype=torch.float32).to(device)
        batch_action = torch.tensor(np.array(batch_action), dtype=torch.int64).unsqueeze(1).to(device)
        batch_reward = torch.tensor(np.array(batch_reward), dtype=torch.float32).unsqueeze(1).to(device)
        batch_next_state = torch.tensor(np.array(batch_next_state), dtype=torch.float32).to(device)
        batch_done = torch.tensor(np.array(batch_done), dtype=torch.float32).unsqueeze(1).to(device)
        current_q = policy_net(batch_state).gather(1, batch_action)
        with torch.no_grad():
            next_q = target_net(batch_next_state).max(1)[0].unsqueeze(1)
        expected_q = batch_reward + gamma * next_q * (1 - batch_done)
        loss = F.smooth_l1_loss(current_q, expected_q)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for episode in range(num_episodes):
        total_rewards = [0] * num_products
        states = env.reset()  # shape: (num_products, 4)
        for t in range(max_steps):
            actions = []
            for i in range(num_products):
                a = select_action(states[i], i, epsilon)
                actions.append(a)
            next_states, rewards, done, _ = env.step(actions)
            for i in range(num_products):
                product_one_hot = np.zeros(num_products)
                product_one_hot[i] = 1
                combined_state = np.concatenate((states[i], product_one_hot))
                combined_next_state = np.concatenate((next_states[i], product_one_hot))
                buffer.push(combined_state, actions[i], rewards[i], combined_next_state, done)
                total_rewards[i] += rewards[i]
            states = next_states
        for i in range(num_products):
            rewards_history[i].append(total_rewards[i])
        optimize_model()
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        ax.clear()
        for i in range(num_products):
            ax.plot(range(len(rewards_history[i])), rewards_history[i], label=f'Product {i} Reward')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.legend()
        plot_path = os.path.join(settings.BASE_DIR, 'statics', 'images', 'dqn_reward.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.draw()
        plt.pause(0.01)
    plt.ioff()
    plt.show()
    torch.save(policy_net.state_dict(), 'dqn_policy_net.pth')

'''







    