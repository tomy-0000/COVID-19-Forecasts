# Net
|   | 特徴量 | パラメータ |
| ---- | ---- | ---- |
| net1 | カウント | hidden_size: 32 <br> num_layers: 1 |
| net2 | カウント <br> 曜日(one hot encoding) | hidden_size: 32 <br> num_layers: 1 |
| net3 | カウント <br> 曜日(Embedding) | weather_embedding_dim: 8 <br> hidden_size: 32 <br> num_layers: 1 |
| net4 | カウント <br> 気温 <br> 降水量 <br> 風速 <br> 現地気圧 <br> 相対温度 <br> 蒸気圧 <br> 天気 <br> 雲量 | hidden_size: 32 <br> num_layers: 1 |