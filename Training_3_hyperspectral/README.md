## Training_3_hyperspectral   孙采萱-398 

实训题目：9个类别的高光谱数据分类

使用模型：svc

预测结果：result.csv 

#### 1.将.mat文件转换为.csv

`dataFile = '/Users/Administrator/Desktop/项目实训/Schoolwork_2/9-train/data2_train.mat'`
`data = sio.loadmat(dataFile)`
`print(data)`
`data1 = data['data2_train']`
`dfdata = pd.DataFrame(data1)`
`datapath1 ='/Users/Administrator/Desktop/项目实训/Schoolwork_2/train/01.csv'`
`dfdata.to_csv(datapath1,index = False)`

//所有的.mat都按照这种方式进行转换

#### 2.合并csv文件并进行规范化处理

```
data =np.vstack((data1,data2,data3,data4,data5,data6,data7,data8,data9))

#标准化数据并存储
data_D = preprocessing.StandardScaler().fit_transform(data[:,:-1])
data_L = data[:,-1]

# 将结果存档后续处理
import pandas as pd
new = np.column_stack((data_D,data_L))
new_ = pd.DataFrame(new)
new_.to_csv('train1/KSC.csv',header=False,index=False)
```

#### 3.划分训练集和测试集，训练模型并进行预测

```
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

#测试
testdata = np.genfromtxt('train/KSC.csv',delimiter=',')
data_test = testdata[:,:-1]
label_test = testdata[:,-1]

clf = joblib.load("KSC5.m")
predict_label = clf.predict(data_test)
accuracy = metrics.accuracy_score(label_test, predict_label)*100
print(accuracy)
```

//通过参数调优，保存了十个模型，正确率在94.6531%—95.9537%之间，本次提交的测试结果使用模型是KSC1_MODEL.m

#### 4.导入测试集，输出预测结果

```
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
```

##### 分析：本次作业任务完成，训练十个模型，选取其中一个进行结果预测，预测结果存入result.csv文件中