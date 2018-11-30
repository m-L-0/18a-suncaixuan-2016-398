## TensorFlow Schoolwork_1  孙采萱-398

使用TensorFlow设计K近邻模型，并使用鸢尾花数据集训练、验证模型 

####  1.将鸢尾花数据集安装8 : 2的比例划分成训练集与验证集 

`Xtrain, Xtest, Ytrain, Ytest = train_test_split(iris1, iris['Y'], test_size=0.20,random_state=40)`

#### 2.设计、训练、预测

    #运行会话，训练模型
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
#### 3.模型预测准确率

![](G:\18a-suncaixuan-2016-398\TensorFlow Schoolwork_1\image\pre.png)

##### 分析：本次项目任务完成