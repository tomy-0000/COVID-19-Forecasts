# COVID-19-Forecasts
COVID-19の感染者数を予測

# Usage
```nets/```にnetを定義  
訓練・予測・maeを保存するため```train_pred.py```を実行 引数として上で定義したnetのファイル名を拡張子抜きで指定(複数指定可)  
```
python train_pred.py net1 net2
```
maeを比較するため```compare.py```を実行 引数として上で定義したnetのファイル名を拡張子抜きで指定(複数指定可) 
```
python compare.py net1 net2
```

# Net
|   | 特徴量 | パラメータ |
| ---- | ---- | ---- |
| net1 | カウント | hidden_size <br> num_layers |
| net2 | カウント <br> 曜日(one hot encoding) | hidden_size <br> num_layers |
| net3 | カウント <br> 曜日(Embedding) | day_embedding_dim <br> hidden_size <br> num_layers |
| net4 | カウント <br> 気温 <br> 降水量 <br> 風速 <br> 現地気圧 <br> 相対温度 <br> 蒸気圧 <br> 天気 <br> 雲量 | hidden_size <br> num_layers |
| net5 | カウント <br> 曜日(Embedding) <br> 気温 <br> 降水量 <br> 風速 <br> 現地気圧 <br> 相対温度 <br> 蒸気圧 <br> 天気 <br> 雲量 | hidden_size: 32 <br> num_layers: 1 |
| net6 |  <br> カウント <br> 緊急事態宣言(経過日数) | hidden_size: 32 <br> num_layers: 1 |
| net7 | 【移動平均】 <br> カウント | hidden_size: 32 <br> num_layers: 1 |
| net8 | 【移動平均】 <br> カウント <br> 気温 <br> 降水量 <br> 風速 <br> 現地気圧 <br> 相対温度 <br> 蒸気圧 <br> 天気 <br> 雲量 | hidden_size: 32 <br> num_layers: 1 |
| net9 | 【移動平均】 <br> カウント <br> 緊急事態宣言(経過日数) | hidden_size: 32 <br> num_layers: 1 |
| net10 | 【移動平均】 <br> カウント <br> 気温 <br> 降水量 <br> 風速 <br> 現地気圧 <br> 相対温度 <br> 蒸気圧 <br> 天気 <br> 雲量 <br> 緊急事態宣言(経過日数) | hidden_size: 32 <br> num_layers: 1 |
| net11 | 【移動平均】 <br> カウント <br> 緊急事態宣言(binary) | hidden_size: 32 <br> num_layers: 1 |

# 参考リンク
[COVID-19 Challenge - SIGNATE](https://signate.jp/covid-19-challenge/)

[気象庁](https://www.data.jma.go.jp/gmd/risk/obsdl/)
