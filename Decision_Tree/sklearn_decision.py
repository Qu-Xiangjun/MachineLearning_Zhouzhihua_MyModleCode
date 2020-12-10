"""
@time: 2020-10-15
@author: Qu Xiangjun 20186471
@describe: 基于sklearn库的决策树模块对西瓜数据集2.0(周志华《机器学习》)二分类
"""
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from DT_main import load_dataset
import numpy as np
import pandas as pd

def load_dataset():
    """
    从asv表格导入数据
    @return x 训练数据 shape=[6,17] ,
            y 训练数据的结果 shape =[1,17],
            attribute 属性list，包含字典{属性名称：取值} 
            eg:[{'色泽': ['浅白', '青绿', '乌黑']}, {'根蒂': ['蜷缩', '硬挺', '稍蜷']}]
    """
    filename = 'E:\\Study\\Term5\\Merchine Learning\\Lab\\Lab02\\xigua2.0.csv'
    data = pd.read_excel(filename)
    # 7行17列的矩阵，第一行为色泽，第二行为根蒂，
    # 第三行为敲声，第四行为纹理，第五行为脐部，第柳行为触感
    x = [list(data[u'色泽']), list(data[u'根蒂']),
        list(data[u'敲声']), list(data[u'纹理']),
        list(data[u'脐部']), list(data[u'触感'])]
    # 1行17列，好瓜结果，1为好瓜，0为坏瓜
    y = list(data[u'好瓜'])
    # 属性：取值
    attribute = []
    attribute_name = ['色泽','根蒂','敲声','纹理','脐部','触感']
    for i in range(len(x)):
        label = set(x[i])
        lable_list = list(label)
        dic = {}
        dic[attribute_name[i]] = lable_list
        attribute.append(dic)
    for i in range(len(attribute)):
        data_ = x[i]
        temp = 0
        for k in attribute[i]:
            for v in attribute[i][k]:
                for j in range(len(data_)):
                    if(data_[j] == v):
                        data_[j] = temp
                temp += 1
    
    return x,y

if __name__ == "__main__":
    x,y = load_dataset()
    x = np.array(x)
    y = np.array(y)
    x = x.T
    dtc = DecisionTreeClassifier()
    dtc.fit(x, y)
    result = dtc.predict(x)
    print(y)
    print(result)