# %% 必要なライブラリのインポート
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import os
from sklearn.preprocessing import MinMaxScaler

# %% データディレクトリの設定
def get_data_dir():
    """データディレクトリのパスを取得する"""
    # スクリプトのディレクトリを取得
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # データディレクトリのパスを構築
    data_dir = os.path.join(script_dir, 'input_data/')
    return data_dir

# %% データディレクトリの確認
DATA_DIR = get_data_dir()
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"データディレクトリが見つかりません: {DATA_DIR}\n現在の作業ディレクトリ: {os.getcwd()}")

eth_path = os.path.join(DATA_DIR, "merged_eth.csv")
eth_df = pd.read_csv(eth_path)
print(eth_df.head())

# %% 正規化
target_data = eth_df.filter(["Close"])
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_values = scaler.fit_transform(target_data)
y_data = pd.DataFrame(scaled_values, index=target_data.index, columns=target_data.columns)
print(y_data.head())

feature_data = eth_df.filter(["Open", "High", "Low"])
scaled_values = scaler.fit_transform(feature_data)
x_data = pd.DataFrame(scaled_values, index=feature_data.index, columns=feature_data.columns)
print(x_data.head())

x_train, x_test, y_train, y_test  = train_test_split(x_data, y_data, test_size=0.3, random_state=42)
print(f'x_train shape: {x_train.shape}, y_train shape: {y_train.shape}')
print(f'x_test shape: {x_test.shape}, y_test shape: {y_test.shape}')

# %% モデルの訓練
model = LGBMRegressor(n_estimators=10000, learning_rate=0.01, random_state=42)
model.fit(x_train, y_train)

# %% 予測と評価
predictions = model.predict(x_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f'RMSE: {rmse}')