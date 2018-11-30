# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 08:06:56 2018

@author: suncaixuan398

使用TensorFlow设计K近邻模型，并使用鸢尾花数据集训练、验证模型

"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


#这里使用本地数据集作为测试，以下是导入数据集代码 
iris = pd.read_csv('iris.csv')
iris1= iris.drop('Y', 1)


Xtrain, Xtest, Ytrain, Ytest = train_test_split(iris1, iris['Y'], test_size=0.20,random_state=40)

train_data = np.array(Xtrain)#np.ndarray()
text_data = np.array(Xtest)
Xtrain=train_data.tolist()#list
Xtest=text_data.tolist()

Ytest = list(Ytest)
Ytrain = list(Ytrain)
# 输入占位符
xtr = tf.placeholder("float", [None, 4])
xte = tf.placeholder("float", [4])


# 计算L1距离
distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
# 获取最小距离的索引
pred = tf.arg_min(distance, 0)

#分类精确度
accuracy = 0.

# 初始化变量
init = tf.global_variables_initializer()

# 运行会话，训练模型
with tf.Session() as sess:
    # 运行初始化
    sess.run(init)

    # 遍历测试数据
    for i in range(len(Xtest)):
        # 获取当前样本的最近邻索引
        nn_index = sess.run(pred, feed_dict={xtr: Xtrain, xte: Xtest[i]})   #向占位符传入训练数据
        # 最近邻分类标签与真实标签比较
       # print("Test", i, "Prediction:", np.argmax(Ytrain[nn_index]), "True Class:", np.argmax(Ytest[i])) 
        print("Sample", i, " - 预测:", Ytrain[nn_index], "正确:", Ytest[i])
        # 计算准确率
        if Ytrain[nn_index] == Ytest[i]:
            accuracy += 1. / len(Xtest)

print("Done!")
print("Accuracy:", accuracy)
