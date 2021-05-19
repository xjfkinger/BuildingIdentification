# BuildingIdentification
利用语义分割进行地表建筑物识别，将地表航拍图像素划分为有建筑物和无建筑物两类

数据：
train_mask.csv：存储图片的标注的rle编码
train和test文件夹：存储训练集和测试集图片训练集有30000张512*512图片，测试集2500张图片

利用Fcn+ResNet50网络模型进行训练
