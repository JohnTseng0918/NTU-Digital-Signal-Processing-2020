# Bearing health condition classification 軸承健康狀態分類
## 版本環境
- torch             1.7.1
- numpy             1.19.4 
- pandas            1.1.5 

## 檔案說明
- train.py
  - training and validation
  - 會產生model.pth
  - 有 --epoch 可以使用
    - 預設epoch為100
- inference.py
  - 做為inference使用，會載入model和data，並且產生myAns.csv檔

## 資料說明
- 資料請放在Data資料夾中才能執行

## 程式執行方式
    python train.py
    python inference.py