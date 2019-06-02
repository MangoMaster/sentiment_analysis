# 情感分析

## 作者

魏家栋，计72班，学号2017011445

## 运行环境

Python3 (Python 3.5.2)
所需Python3库有：
- Keras 2.2.4
- TensorFlow 1.13.1
- sklearn 0.21.2

## 使用方法

若只需要计算模型的评价指标，运行：
```sh
cd ./src
python3 ./test_model.py
```
若希望预处理数据或训练模型，请将给定的新浪新闻和词向量(https://cloud.tsinghua.edu.cn/f/7928cb6c3db34c67b1b0/ )放入data文件夹下，然后运行：
```sh
cd ./src
python3 ./bake_data.py
python3 ./mlp.py
```

## 目录层次

- doc文件夹：程序[文档](./doc/情感分析作业报告.pdf)。
- src文件夹：情感分析神经网络程序的源代码，内含bake_data.py（数据预处理）、test_model.py（计算评价指标等）和各类神经网络模型的源代码。
- data文件夹：输入的新浪新闻文件、词向量文件，以及预处理产生的文件等。由于占用空间过大，未附在压缩包中。
- models文件夹：各类神经网络已经训练好的模型。
