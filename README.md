# DSAI-HW2-2021

## 使用說明
安裝相依套件
```
pip install -r requirements.txt
```
運行主程式
```
python app.py --training training_data.csv --testing testing_data.csv --output output.csv
```

## 模型說明

### LSTM

#### 模型介紹
長短期記憶模型，主要用於時間序列資料的預測，可以把連續的資料集切隔成一個個連續的窗格來進行趨勢預測，本專案將窗格大小設定為 6，這意味著可運用前六日的資料來預測隔日的資料。

#### 資料準備
使用IBM股票的歷史資料來當訓練集，資料欄位依序分別為「開盤價」、「當日最高價」、「當日最低價」、「收盤價」，並額外添設一個「中間值」(當日最高價與當日最低價的平均)的資料欄位。
本專案使用「開盤價」最為模型所需的「ytrain欄位」，而把每日資料欄位切割成一個一個窗格作為模型所需的「xtrain欄位」，再放入模型進行訓練。

#### 參數設定
* Batch_size設定為 32
* Epochs設定為100
* Kernel_initializer使用glorot_normal


#### 訓練結果
![](https://imgur.com/RswW2zt)
