"""
@time: 2020-10-10
@author: Qu Xiangjun 20186471
@describe: Logistic 二分类算法的实现
"""
import numpy as np
from numpy import linalg

def Logistic_train(x,y,beta):
    """
    利用牛顿迭代法求解
    @x 训练数据x shape=(a,b)
    @y 训练数据y shape=(b,)
    @beta 系数初始值 shape=(a,)
    @return beta，系数的迭代结果
    """
    step = 0  # 迭代次数
    # 定义最小迭代变化量
    ep = 0.000001

    while 1:
        # 对β进行转置取第一行
        # 再与x相乘（dot）,beta_x表示β转置乘以x)
        beta_x = np.dot(beta.T[0], x)
        # 先求关于β的一阶导数和二阶导数
        dbeta = 0
        d2beta = 0
        for i in range(len(x[0])):
            # 一阶导数
            dbeta = dbeta - np.dot(np.array([x[:, i]]).T,
                                    (y[i] - (np.exp(beta_x[i]) / (1 + np.exp(beta_x[i])))))
            # 二阶导数
            d2beta = d2beta + np.dot(np.array([x[:, i]]).T, np.array([x[:, i]]).T.T) * (
                        np.exp(beta_x[i]) / (1 + np.exp(beta_x[i]))) * (
                                    1 - (np.exp(beta_x[i]) / (1 + np.exp(beta_x[i]))))
        #得到牛顿方向
        d = - np.dot(linalg.inv(d2beta), dbeta)
        # 迭代终止条件
        if np.linalg.norm(d) <= ep:  # 牛顿方向向量的模小于0.000001时认为很接近极小值点，停止迭代
            break  # 满足条件直接跳出循环
        # 牛顿迭代法更新β
        beta = beta + d
        step = step + 1

    print('系数')
    for i in beta:
        print(i)
    print('迭代次数：', step)
    
    return beta

# sigmod 函数
def sigmod(x):
    return 1/(1+np.exp(-x))

"""
随机梯度上升算法
@x 训练数据x shape=(m,n)
@y 训练数据y shape=(m,)
@beta 系数初始值 shape=(n,)
@numIter 迭代次数
@return beta，系数的迭代结果
"""
def stocGradAscent(train_x,train_y,numIter=150):
    m,n=np.shape(train_x)
    beta=np.ones(n)
    for j in range(numIter):  
        dataIndex=list(range(m))
        for i in range(m):
            alpha=4/(1.0+j+i)+0.01
            randIndex=int(np.random.uniform(0,len(dataIndex)))
            h=sigmod(sum(train_x[randIndex]*beta))
            error=train_y[randIndex]-h
            beta=beta+alpha*error*train_x[randIndex]
            del(dataIndex[randIndex])
    print('系数')
    for i in beta:
        print(i)
    
    return beta


"""
@x 测试数据x shape = (a,b)
@beta 系数值 shape = (a,)
@return 测试结果y（list） shape = (b,)
"""
def Logistic_test(test_x,beta):
    temp = np.dot(beta, test_x)
    y = 1/(1+np.exp(-temp))
    for i in range(len(y)):
        if(y[i]<0.5):
            y[i] = 0
        else:
            y[i] = 1
    print("test_y:",y)
    return y