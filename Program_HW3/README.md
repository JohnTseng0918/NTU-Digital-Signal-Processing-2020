# Image Inpainting using Linear Autoencoder
## 版本環境

- torch             1.7.1     
- torchvision       0.8.2     
- Pillow            8.0.1     

## 程式執行說明
    python train.py --num 0,1 --epoch 20
    python inference.py --num 0,1 --pic 10

### 執行說明
- 執行完train.py會產生autoencoder.pth的model
- 執行完inference.py會產生圖片

### 檔案說明
- train.py: train的檔案
- inference.py: test的檔案
- autoencoder.py: autoencoder的架構
- utils.py: get_target和postprocessing放這裡

### 指令說明
- train.py
  - --num 0,1 指的是只用數字0,1兩個類別
  - --epoch 20 表示訓練20個epoch
- inference.py
  - --num 0,1 指的是只用數字0,1兩個類別
  - --pic 10 表示輸出 10*3張圖片(原始,污染,修復後)