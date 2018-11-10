# Tensorflow_NIMA_Food_Photo_Aesthetic_Evaluation
透過Tensorflow Slim實作Google NIMA論文,  透過VGG16模型幫美食照片進行評分.
5分以上代表high, 低於5分代表low</br>
<img src="https://github.com/masterTW/Tensorflow_NIMA_Food_Photo_Aesthetic_Evaluation/blob/master/photo1.png?raw=true" height=60% width=60%>
# User Story:
  - user想寫一篇受歡迎的食記, 拍了一堆品質參差不齊的美食照片卻不知道使用哪一張, 透過本系統從美食照片中篩選出最吸引人的食物照片, 供user參考.

# Introduction:
- Google NIMA論文[1]透過AVA美學圖片資料集[3], 對圖片進行美學評分, 論文[2]顯示預測美食照片的分數,只需要少量的AVA美食照片做訓練即可有不錯的結果, 因此, 本專案使用5000張AVA的美食照片當作資料集.
# Requirements:
  - Python 3
  - TensorFlow

# Training:
1. 下載 [AVA資料集](https://github.com/mtobeiyf/ava_downloader)
2. 下載 [Slim VGG16 pretrained model](https://github.com/tensorflow/models/tree/master/research/slim). These CNNs have been trained on the ILSVRC-2012-CLS image classification dataset.
3. I will release the training code soon.
# Evaluation:
1. 下載 [model](https://drive.google.com/file/d/16eK7ByJi1zV68v7OS6LKshDlll-AeSpj/view?usp=sharing) 或自己訓練model
2. 執行下列指令, 程式會去讀取資料夾中的所有圖片進行美學評分<br />

```python3 evaluate_nima_vgg16.py --photo_dir= <path to photo> --vgg16_path= <path to vgg16>```

# Resullt:
  - 美食照片訓練data的分類成功率達到88.3%, 測試資料的分類準確度達到73.5%, 符合論文[2]實驗結果.
# To-Do List:
  - 縮小模型檔案大小, 使用MobileNet
  - 用Tensorflow.js製作Web版本
# References:
 1.   Talebi, Hossein, and Peyman Milanfar. "NIMA: Neural Image Assessment" IEEE Transactions on Image Processing, 2017
 2.   Jiayu Lou, Hang Yang. "Food Image Aesthetic Quality Measurement by Distribution Prediction", 2018
 3.   Naila Murray, Luca Marchesotti, Florent Perronnin. "AVA: A Large-Scale Database for Aesthetic Visual Analysis", 2012
