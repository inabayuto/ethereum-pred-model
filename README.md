# Ethereum価格予測モデル

## 概要
このリポジトリは、イーサリアム（ETH/USD）の価格予測を行うための機械学習モデルの実装例です。LSTM（深層学習）とLightGBM（勾配ブースティング）を用いた2種類のアプローチを提供しています。

## ファイル構成
- `baseline.py`  
  LSTM（長短期記憶）ニューラルネットワークを用いた時系列予測モデル。
  - データの前処理（複数取引所データのマージ・正規化）
  - LSTMモデルの構築・学習・保存
  - 予測結果の可視化
- `lightgbm_model.py`  
  LightGBMを用いた回帰モデル。
  - データの前処理（特徴量抽出・正規化）
  - LightGBMモデルの学習・評価
- `requirements.txt`  
  必要なPythonパッケージ一覧
- `input_data/`  
  入力データ（例：ETH_1min.csv, Bitstamp_ETHUSD_1h.csv, merged_eth.csv）
- `output_data/`  
  学習済みモデルや予測結果の保存先

## 実行環境について
- 本リポジトリの学習処理は計算量が多く、CPU環境では非常に時間がかかる場合があります。
- 可能であれば、GPU環境（Google Colab等）での実行を推奨します。
- Google Colabでの実行例やセットアップ方法も今後追加予定です。

## セットアップ方法
1. 必要なパッケージをインストール
   ```
   pip install -r requirements.txt
   ```
2. `input_data/`ディレクトリに必要なデータ（CSV）を配置
3. 各スクリプトを実行

## 簡単な流れ
1. 取引所ごとのETH価格データをマージ・前処理
2. LSTMまたはLightGBMで学習・予測
3. 予測精度の評価・グラフ表示
4. 学習済みモデルや予測結果を保存 

## 参考
- [LightGBMで株価予測](https://qiita.com/MuAuan/items/3e876ac94b5600448912)
- [LSTMによる株価予測](https://qiita.com/pyman123/items/70406028c7607102ad83)
- [イーサリアムデータ取得](https://aifx.tech/ethereum%E3%81%AE%E9%81%8E%E5%8E%BB%E3%83%87%E3%83%BC%E3%82%BF%E3%82%92%E3%83%80%E3%82%A6%E3%83%B3%E3%83%AD%E3%83%BC%E3%83%89%E3%81%99%E3%82%8B/)
- [GoogleColabの使い方まとめ](https://qiita.com/shoji9x9/items/0ff0f6f603df18d631ab)
