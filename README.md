# Tensorflow_NIMA_Food_Photo_Aesthetic_Evaluation
Implementation of Google NIMA paper by Tensorflow Slim. Evaluate food photos with VGG16 model. Score of 5 or above means successful photo. Lower than 5 means not so good.</br>
<img src="https://github.com/masterTW/Tensorflow_NIMA_Food_Photo_Aesthetic_Evaluation/blob/master/photo1.png?raw=true" height=60% width=60%>
# User Story:
  - User wants to publish a popular food journal, but does not know which food photos to choose from a wide range of photos. This system helps user to select the most appealing food photos for user’s reference. 

# Introduction:
- Google NIMA paper[1] mentions evaluating photos esthetically with AVA esthetic photo gallery[3]. Another paper[2] shows that to train a model to evaluate food photos, one only needs to use a few AVA food photos to have successful results. So this project uses 5000 AVA food photos as dataset.
# Requirements:
  - Python 3
  - TensorFlow

# Training:
1. Download [AVA dataset](https://github.com/mtobeiyf/ava_downloader)
2. Download [Slim VGG16 pre-trained model](https://github.com/tensorflow/models/tree/master/research/slim). These CNNs have been trained on the ILSVRC-2012-CLS image classification dataset.
3. Training code will be released soon.
# Evaluation:
1. Download [model](https://drive.google.com/file/d/16eK7ByJi1zV68v7OS6LKshDlll-AeSpj/view?usp=sharing)  or self-trained model.
2. Run below instruction and the program will load all the photos in dataset for esthetic evaluation. <br />

```python3 evaluate_nima_vgg16.py --photo_dir= <path to photodir> --vgg16_path= <path to vgg16>```

# Resullt:
  - Esthetic score of 5 or above means successful photo. Lower than 5 means not so good. Tested the model with 500 AVA food photos and confirmed the accuracy is up to 73.5%, which matches the result of the paper[2].
# To-Do List:
  - Downsize the file of model with MobileNet
  - Create Web version with Tensorflow.js
# References:
 1.   Talebi, Hossein, and Peyman Milanfar. "NIMA: Neural Image Assessment" IEEE Transactions on Image Processing, 2017
 2.   Jiayu Lou, Hang Yang. "Food Image Aesthetic Quality Measurement by Distribution Prediction", 2018
 3.   Naila Murray, Luca Marchesotti, Florent Perronnin. "AVA: A Large-Scale Database for Aesthetic Visual Analysis", 2012
