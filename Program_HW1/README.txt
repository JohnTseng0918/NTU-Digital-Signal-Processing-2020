版本環境
python 3.8.5
pandas 1.1.3
numpy 1.19.2
Pillow 8.0.0

第一題作業
python ave.py
會產生所有資料的平均，以及所有個別數字資料的平均

第二題作業
python plot_eigenvector.py
會產生所有資料的前三大eigenvector的圖片，以及個別數字前三大eigenvector的圖片

第三、四題作業
python PCA.py
後面有參數可以下
--center 0 表示non-centered PCA
--center 1 表示centered PCA
--num 後面可以接數字，請用逗點隔開，例如 --num 1,2,3,4,5
就可以針對12345的資料一起做PCA

預設參數列表如下
python PCA.py --center 1 --num 0,1,2,3,4,5,6,7,8,9