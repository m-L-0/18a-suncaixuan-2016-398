## Clustering Schoolwork_1   孙采萱-398 

以谱聚类或者马尔科夫聚类对**鸢尾花数据集**进行处理，并输出正确率。 

#### 1.将鸢尾花数据画成图的形式

​     将鸢尾花的数据按照其分类，在图上表示出来，得到如下效果

![](G:\18a-suncaixuan-2016-398\Clustering Schoolwork_1\image\iris.png)

#### 2.求取带权邻接矩阵

    def myKNN(S, k, sigma=2.0):
        N = len(S)
        A = np.zeros((N,N))
    for i in range(N):
        dist_with_index = zip(S[i], range(N))
        dist_with_index = sorted(dist_with_index, key=lambda x:x[0])
        neighbours_id = [dist_with_index[m][1] for m in range(k+1)] # xi's k nearest neighbours
    
        for j in neighbours_id: # xj is xi's neighbour
            A[i][j] = np.exp(-S[i][j]/2/sigma/sigma)
            A[j][i] = A[i][j] # mutually
    return A
#### 3.根据邻接矩阵进行聚类，聚类结果可视化，得到如下效果

![](G:\18a-suncaixuan-2016-398\Clustering Schoolwork_1\image\pre_iris.png)

#### 4.计算正确率

`score_pure = accuracy_score(lable,pure_kmeans)`
`print(score_pure)`

![](G:\18a-suncaixuan-2016-398\Clustering Schoolwork_1\image\pre.png)

准确率89%的原因：鸢尾花数据集存在噪声点，谱聚类适用于球形分，

##### 分析：本次作业任务基本完成，除根据邻接矩阵绘制无向图外，全部完成。