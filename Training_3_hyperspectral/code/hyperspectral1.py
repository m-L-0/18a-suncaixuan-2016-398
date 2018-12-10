# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 16:58:35 2018

@author: 孙采萱-398
实训题目：9个类别的高光谱数据分类
使用模型：SVC
"""

import joblib
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
from sklearn import preprocessing
import pandas as pd
import scipy.io as sio

#训练集按照以下形式将.mat转换为.csv
'''
dataFile = '/Users/Administrator/Desktop/项目实训/Schoolwork_2/9-train/data2_train.mat'
data = sio.loadmat(dataFile)
print(data)
data1 = data['data2_train']
dfdata = pd.DataFrame(data1)
datapath1 ='/Users/Administrator/Desktop/项目实训/Schoolwork_2/train/01.csv'
dfdata.to_csv(datapath1,index = False)
'''

#将转换后的数据进行处理，添加标签，合并9类别数据
data1 = np.array(pd.read_csv('train1/01.csv'))
data2 = np.array(pd.read_csv('train1/02.csv'))
data3 = np.array(pd.read_csv('train1/03.csv'))
data4 = np.array(pd.read_csv('train1/04.csv'))
data5 = np.array(pd.read_csv('train1/05.csv'))
data6 = np.array(pd.read_csv('train1/06.csv'))
data7 = np.array(pd.read_csv('train1/07.csv'))
data8 = np.array(pd.read_csv('train1/08.csv'))
data9 = np.array(pd.read_csv('train1/09.csv'))
data = np.vstack((data1,data2,data3,data4,data5,data6,data7,data8,data9))

#标准化数据并存储
data_D = preprocessing.StandardScaler().fit_transform(data[:,:-1])
data_L = data[:,-1]

# 将结果存档后续处理
import pandas as pd
new = np.column_stack((data_D,data_L))
new_ = pd.DataFrame(new)
new_.to_csv('train1/KSC.csv',header=False,index=False)

# 导入数据集切割训练与测试数据
data = pd.read_csv('train1/KSC.csv',header=None)
data = data.as_matrix()
data_D = data[:,:-1]
data_L = data[:,-1]
data_train, data_test, label_train, label_test = train_test_split(data_D,data_L,test_size=0.2)

# 模型训练与拟合
clf = SVC(kernel='rbf',gamma=0.0045,C=35,random_state=2)
clf.fit(data_train,label_train)
pred = clf.predict(data_test)
accuracy = metrics.accuracy_score(label_test, pred)*100
print(accuracy)

# 存储结果学习模型，方便之后的调用
joblib.dump(clf, "KSC1_MODEL.m")
'''
#测试
testdata = np.genfromtxt('train/KSC.csv',delimiter=',')
data_test = testdata[:,:-1]
label_test = testdata[:,-1]

clf = joblib.load("KSC5.m")
predict_label = clf.predict(data_test)
accuracy = metrics.accuracy_score(label_test, predict_label)*100
print(accuracy)
'''
#导入测试集
#将.mat转换为.csv
dataFile = '/Users/Administrator/Desktop/项目实训/Schoolwork_2/data_test_final.mat'
data = sio.loadmat(dataFile)
print(data)
data1 = data['data_test_final']
dfdata = pd.DataFrame(data1)
datapath1 ='/Users/Administrator/Desktop/项目实训/Schoolwork_2/data_test.csv'
dfdata.to_csv(datapath1,index = False)

#规范化测试集
data_test0=open('/Users/Administrator/Desktop/项目实训/Schoolwork_2/data_test.csv')
data_test=pd.read_csv(data_test0)
data_D = preprocessing.StandardScaler().fit_transform(data_test)

#进行预测
fx = joblib.load("KSC1_MODEL.m")

mx1 = fx.predict(data_D)
import csv
with open("D:/result.csv","w") as csvfile: 
    writer = csv.writer(csvfile)
    writer.writerow(mx1)
    csvfile.close()