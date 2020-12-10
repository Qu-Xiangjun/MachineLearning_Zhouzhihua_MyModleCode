"""
@time: 2020-10-10
@author: Qu Xiangjun 20186471
@describe: 利用自定义的对率回归分类算法对西瓜数据集二分类
"""
import numpy as np
from numpy import linalg
import pandas as pd
from Logistic import Logistic_train,Logistic_test

def get_data():
    # 读取数据
    filename = 'C:\\Users\\49393\\Desktop\\ML_pre\\实验1-屈湘钧-20186471-2018计科卓越\\xigua.csv'
    data = pd.read_excel(filename)
    # 2行17列的矩阵，第一行为密度，第二行为含糖率；加入一行1作为第三行，对应于常数项b
    x = np.array(
        [list(data[u'密度']), list(data[u'含糖率']), [1 for i in range(17)]])
    y = np.array(list(data[u'好瓜']))

    # print("data_x:",x)
    # print("data_y:",y)
    
    # 定义初始值,分别对应密度、含糖率与常数项的系数
    beta = np.array([[0], [0], [1]])  # β列向量
    
    return x,y,beta

    

if __name__ == "__main__":
    x,y,beta = get_data()
    beta = Logistic_train(x,y,beta)
    # 由于西瓜数据集只有17个数据，无法分类做训练与测试，故进行封闭测试
    predict = Logistic_test(x,beta.T[0])
    correct = 0
    for i in range(len(predict)):
        if(y[i]== predict[i]):
            correct += 1
    print("正确率：%.4f"%(correct/len(predict)))
    print("predict value:",predict)
    print("test value:",y)


