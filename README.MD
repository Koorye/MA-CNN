# MA-CNN的复现

## 说明
Backbone采用ResNet18的前两层，否则CIFAR10的64x64图片输入后得到的特征图宽高仅为1，这样的特征图无法使用Multi Attention模块进行前向传播和损失计算。

如要修改，可在weights/ma_cnn.py中更换backbone。
