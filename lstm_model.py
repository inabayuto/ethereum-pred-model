# %% ライブラリのインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import warnings
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error

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

# データの読み込み
ka_eth_path = os.path.join(DATA_DIR, "ETH_1min.csv")
bt_eth_path = os.path.join(DATA_DIR, "Bitstamp_ETHUSD_1h.csv")

# %% データのマージ・保存
def merge_and_save_eth_data(ka_path, bt_path, output_path):
    """
    2つのETHデータを読み込み、カラム名を統一し、縦結合・重複削除して保存する関数
    ka_path: kaikoデータのパス
    bt_path: bitstampデータのパス
    output_path: 保存先ファイルパス
    """
    # データ読み込み
    ka_eth_df = pd.read_csv(ka_path, index_col=0, usecols=[1,2,3,4,5,6], header=0)
    bt_eth_df = pd.read_csv(bt_path, index_col=0, usecols=[1,2,3,4,5,6], header=1)
    # カラム名統一
    ka_eth_df.columns = ['Symbol', 'Open', 'High', 'Low', 'Close']
    bt_eth_df.columns = ['Symbol', 'Open', 'High', 'Low', 'Close']
    # 縦結合
    eth_df = pd.concat([ka_eth_df, bt_eth_df], axis=0)
    # 重複削除（ka_eth_df優先）
    eth_df = eth_df[~eth_df.index.duplicated(keep='first')]
    # 保存先ディレクトリがなければ作成
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # CSV保存
    eth_df.to_csv(output_path)
    print(f"保存完了: {output_path}")
    return eth_df

# 関数を使ってデータをマージ・保存
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input_data/merged_eth.csv')
eth_df = merge_and_save_eth_data(ka_eth_path, bt_eth_path, output_path)
print(f"ユニークなインデックスの数: {eth_df.index.nunique()}, データの数: {len(eth_df)}")


# %%レートの表示   
plt.figure(figsize=(12, 5))
plt.plot(pd.to_datetime(eth_df.index),eth_df["Open"],alpha=0.6,label="eth/usd")
plt.yscale("log")
plt.grid()
plt.legend()
plt.xlabel("時刻", fontname="Hiragino Sans", fontsize=15)
plt.ylabel("ETHUSD", fontname="Hiragino Sans", fontsize=15)
plt.show()

# %%
eth_path = os.path.join(DATA_DIR, "merged_eth.csv")
eth_df = pd.read_csv(eth_path)
print(eth_df.head())

# %%　Closeコラムのみ抽出
data = eth_df.filter(["Close"])
dataset = data.values
print(dataset)

# %% 正規化
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
print(dataset)

# %% データを訓練データと検証データに分割し、7割が訓練用になるように分割
train_size = int(np.ceil(len(dataset) * 0.7))
print(train_size)

train_data = dataset[0: int(train_size), :]
train_data.shape
# %% 訓練データの取得
x_train = []
y_train = []
time_step = 120
for i  in range(time_step, len(train_data)):
    x_train.append(train_data[i-time_step:i, 0])
    y_train.append(train_data[i, 0])

# 訓練データのreshape
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(x_train.shape, y_train.shape)

# %% LSTMモデル構築
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1))) # 128個のノードを持つLSTM層を追加（return_sequences=Trueは全ての層を返す）
model.add(LSTM(64, return_sequences=False)) # 64個のノードを持つLSTM層を追加(return_sequences=Falseは最後の層のみを返す)
model.add(Dense(25)) # 25個のノードを持つ全結合層を追加
model.add(Dense(1)) # 1個のノードを持つ全結合層を追加

# %% モデルのコンパイル
model.compile(optimizer='adam', loss='mean_squared_error')

# %% モデルの訓練
model.fit(x_train, y_train, epochs=5, batch_size=32)

# %% 検証用データの作成
test_data = dataset[train_size - time_step: , :]
x_test = []
for i in range(time_step, len(test_data)):
    x_test.append(test_data[i-time_step:i, 0])
y_test = dataset[train_size:, :]
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# %% モデルの予測
predicted_price = model.predict(x_test)
predicted_price = scaler.inverse_transform(predicted_price)

# %% 予測結果の表示
y_test = y_test.squeeze()
predicted_price = predicted_price.squeeze()

# NaNのマスク（両方にNaNがない行だけを残す）
mask = ~np.isnan(y_test) & ~np.isnan(predicted_price)

# フィルター適用
y_test_clean = y_test[mask]
predicted_clean = predicted_price[mask]

# RMSE計算
rmse = np.sqrt(mean_squared_error(y_test_clean, predicted_clean))
print(f"Test Score: {rmse}")

# %% モデルの保存
output_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_data')
os.makedirs(output_model_dir, exist_ok=True)
model.save(os.path.join(output_model_dir, 'eth_lstm_model.h5'))
print("モデルを保存しました: output_data/eth_lstm_model.h5")

# %% 予測結果の表示
train = eth_df[:train_size]
test = eth_df[train_size:]
test['Predictions'] = predicted_price

plt.figure(figsize=(16, 6))
plt.title('LSTM Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(test[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()