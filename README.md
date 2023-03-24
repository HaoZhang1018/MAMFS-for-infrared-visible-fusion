# MAMFS-for-infrared-visible-fusion
Code of Infrared and Visible Image Fusion Based on Multiclassification Adversarial Mechanism in Feature Space (基于特征空间多分类对抗机制的红外与可见光图像融合)


````
@article{张浩2023基于特征空间多分类对抗机制的红外与可见光图像融合,
  title={基于特征空间多分类对抗机制的红外与可见光图像融},
  author={张浩 and 马佳义 and 樊凡 and 黄珺 and 马泳},
  journal={计算机研究与发展},
  volume={60},
  number={3},
  pages={690--704},
  year={2023}
}
````

#### running environment :<br>
python=2.7, tensorflow-gpu=1.9.0.

#### Prepare data :<br>
Put training image pairs in the "Train_ir" and "Train_vi", and put test image pairs in the "Test_ir" and "Test_vi" folders folders.

#### To train :<br>
Run "CUDA_VISIBLE_DEVICES=X python train_AE.py" to train the proposed autoencoder. <br>
Run "CUDA_VISIBLE_DEVICES=X python train_Fusion.py" to train the proposed feature fusion metwork. <br>


#### To test :<br>
Run "CUDA_VISIBLE_DEVICES=X python demo.py" to test the trained model.
You can also directly use the trained model we provide.

