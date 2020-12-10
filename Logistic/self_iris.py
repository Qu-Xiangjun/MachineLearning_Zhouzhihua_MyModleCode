"""
@time: 2020-10-10
@author: Qu Xiangjun 20186471
@describe: 利用自定义的对率回归分类算法对iris数据集三分类，一对多方式实现
"""
import numpy as np
from numpy import linalg
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from Logistic import stocGradAscent,Logistic_test

def get_data():
    # 在sklearn的数据集库中导入
    iris = datasets.load_iris()  # 加载莺尾花数据集
    x_train = iris.data
    y_train = iris.target

    x_train,x_test,y_train,y_test = train_test_split(x_train,y_train,test_size = 0.25,random_state=0,stratify = y_train)
    x_train,x_test = x_train.T,x_test.T
    x_train = np.concatenate((x_train,np.array([[1 for i in range(112)]])))
    x_test = np.concatenate((x_test,np.array([[1 for i in range(38)]])))

    # print(x_test)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    
    # 定义初始值,分别对应萼片长度、宽度，花瓣长度、宽度与常数项的系数
    beta = np.array([[0], [0], [0], [0], [1]])  # β列向量
    
    return x_train,x_test,y_train,y_test, beta

    

if __name__ == "__main__":
    x_train,x_test,y_train,y_test,beta = get_data()
    # print(y_test) [0 0 0 0 1 1 1 0 1 2 2 2 1 2 1 0 0 2 0 1 2 1 1 0 2 0 0 1 2 1 0 1 2 2 0 1 2 2]
    
    # ova
    # 将0作为类0，1、2作为类1
    x1_train = x_train
    y1_train = []
    for i in range(len(y_train)):
        if(y_train[i]>0):
            y1_train.append(1)
        else:
            y1_train.append(0)
    y1_train = np.array(y1_train)  # np数组化    
    # print("y1_train",y1_train)
    # 训练第一轮的参数
    beta_first = stocGradAscent(x1_train.T,y1_train,)
    
    
    # ova
    # 将在第一轮的ova中判断为1类的筛选出来，实施第二轮的二分类
    x2_train = []
    y2_train = []
    for i in range(len(y_train)):
        if(y_train[i]==1):
            y2_train.append(0)
            x2_train.append(x_train[:,i])
        elif(y_train[i]==2):
            y2_train.append(1)
            x2_train.append(x_train[:,i])
    y2_train = np.array(y2_train)  # np数组化
    x2_train = np.array(x2_train).T # np数组化
    
    print("y2_train",y2_train)
    print("x2_train",x2_train)
    # 训练第二轮的参数
    beta_second = stocGradAscent(x2_train.T,y2_train)
    
    # 计算错误率
    x1_test = x_test
    y1_test = []
    for i in range(len(y_test)):
        if(y_test[i]>0):
            y1_test.append(1)
        else:
            y1_test.append(0)
    y1_test = np.array(y1_test)
    # print("y1_test",y1_test)
    #获得第一次1对多预测结果
    predict = Logistic_test(x1_test,beta_first) 

    x2_test = []
    y2_test = []
    for i in range(len(predict)):
        if(predict[i]>0):
            x2_test.append(x_test[:,i])
            y2_test.append(y_test[i])
    x2_test = np.array(x2_test).T
    y2_test = np.array(y2_test)
    print("x2_test",x2_test.shape)
    print("y2_test",y2_test.shape)
    # 获得第二次分类的预测结果
    predict_second = Logistic_test(x2_test,beta_second)
    print("predict_second:",predict_second)
    
    j = 0
    y_predict = []  # 真实预测的结果列表
    # 合并第一二次的预测结果
    for i in range(len(predict)):
        if(predict[i]>0):
            y_predict.append(predict_second[j]+1)
            j+=1
        else:
            y_predict.append(predict[i])
    y_predict = np.array(y_predict)

    correct = 0  # 正确个数
    for i in range(len(y_predict)):
        if(y_test[i]== y_predict[i]):
            correct += 1
    
    print("正确率：%.4f"%(correct/len(y_predict)))
    print("predict value:",y_predict)
    print("test value:",y_test)
    
    


