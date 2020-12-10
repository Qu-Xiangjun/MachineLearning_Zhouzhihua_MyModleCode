"""
@time: 2020-10-15
@author: Qu Xiangjun 20186471
@describe: 利用自定义的决策树分类算法对西瓜数据集2.0(周志华《机器学习》)二分类
"""
import numpy as np
from numpy import linalg
from math import log
import copy
import random
import json

def Ent(dataset):
    """
    计算信息熵
    @dataset shape=[x,] 待计算数据
    @return shannon 信息熵
    """
    dictcategory={}
    for i in dataset:
        category=i
        if category not in dictcategory:
            dictcategory[category]=0
        dictcategory[category]+=1
    num=len(dataset)
    shannon=0.0
    for i in dictcategory:
        prob=float(dictcategory[i])/num
        shannon-=prob*log(prob,2)
    return shannon

def Gain(dataset,y): 
    """
    计算信息增益
    @dataset shape=[x,] 全部待计算数据 
        eg：['青绿' '乌黑' '乌黑' '青绿' '浅白' '青绿' '乌黑' '乌黑' '乌黑']
    @y 训练数据labels shape=[x,]
    @return gain 信息增益
    """   
    ent_D = Ent(y) # 根节点信息熵
    # print(ent_D)
    classified = {} # 字典，key为dataset中的类别名称，value为类别对应的信息熵
    for i in dataset:
        if(i not in classified):
            classified[i] = 0
    # print(classified)
    
    # 计算信息增益
    gain = ent_D
    for x in classified:
        dataset_y = [] # dataset 中每一类别的y值
        for j in range(len(dataset)):
            if(dataset[j] == x):
                dataset_y.append(y[j])
        # 计算每一类别的信息熵
        classified[x] = Ent(dataset_y)
        gain -= classified[x] * len(dataset_y) / len(dataset)
    return gain
    
def Gain_ratio(dataset,y):
    """
    计算信息增益率
    @dataset shape=[x,] 全部待计算数据 
        eg：['青绿' '乌黑' '乌黑' '青绿' '浅白' '青绿' '乌黑' '乌黑' '乌黑']
    @y 训练数据labels shape=[x,]
    @return gain_ratio 信息增益率
    """  
    gain = Gain(dataset,y) # 信息增益
    # print(gain)
    classified = {} # 字典，key为dataset中的类别名称，value为类别对应的信息熵
    for i in dataset:
        if(i not in classified):
            classified[i] = 0
        classified[i] += 1

    IV = 0 #固有值初始化
    for x in classified:
        IV -= ( classified[x]/len(dataset) ) * ( log(classified[x]/len(dataset), 2) )
    # print(IV)
    # 计算信息增益率
    gain_ratio = gain / IV
    return gain_ratio

def Gini(dataset):
    """
    计算基尼值
    @dataset shape=[x,] 待计算数据
    @return gini 信息熵
    """
    dictcategory={}
    for i in dataset:
        category=i
        if category not in dictcategory:
            dictcategory[category]=0
        dictcategory[category]+=1
    num=len(dataset)
    gini=1.0
    for i in dictcategory:
        gini -= float(pow(dictcategory[i]/len(dataset),2))
    return gini

def Gini_index(dataset,y): 
    """
    计算基尼指数
    @dataset shape=[x,] 全部待计算数据 
        eg：['青绿' '乌黑' '乌黑' '青绿' '浅白' '青绿' '乌黑' '乌黑' '乌黑']
    @y 训练数据labels shape=[x,]
    @return gini_ratio 信息增益率
    """
    classified = {} # 字典，key为dataset中的类别名称，value为类别对应的数量
    for i in dataset:
        if(i not in classified):
            classified[i] = 0
        classified[i] += 1
        
    gini_index = 0
    for x in classified:
        dataset_y = [] # dataset 中每一类别的y值
        for j in range(len(dataset)):
            if(dataset[j] == x):
                dataset_y.append(y[j])
        # 计算每一类别的信息熵
        classified[x] = Gini(dataset_y)
        gini_index += len(dataset_y) * classified[x] / len(dataset)
    return gini_index


def Best_label(dataset,y,method = 'id3'):
    """
    计算最优属性划分
    @dataset shape=[a,x] 全部待计算数据 a为属性个数，x为数据个数
    @y 训练数据划分类型结果 shape=[x,]
    @method 选择最优划分属性方法，id3 为信息增益法，
            c4.5 为信息增益率法，cart为选择基尼系数法
    @return best_label_index
    """
    assert method in ['id3', 'c4.5','cart'] #"method 须为id3或c45,cart"
    if (method == 'id3'): # 信息增益法
        max_gain = 0
        max_gain_index = 0 # 最大增益的dataset行数下标
        i = 0
        for data in dataset:
            if(Gain(data,y) > max_gain):
                max_gain = Gain(data, y)
                max_gain_index = i
            elif (Gain(data, y) == max_gain):# 若信息增益相同，则随机选择
                rd = random.randint(0,1)
                if(rd==1):
                    max_gain = Gain(data, y)
                    max_gain_index = i
            i+=1
        return max_gain_index
        
    elif (method == 'c4.5'): # 信息增益率法
        gain_list = []
        sum = 0
        for i in dataset:
            gain_list.append(Gain(i, y))
            sum += Gain(i,y)
        average = sum / len(dataset) # 信息增益平均值
        gain_ratio_list = []
        for i in range(len(dataset)):
            if(gain_list[i] >= average ):
                gain_ratio_list.append(Gain_ratio(dataset[i],y))
        max_gain_ratio_index = 0 # 最大增益率的dataset行数下标
        max_gain_ratio = 0
        temp = 0
        for i in gain_ratio_list:
            if(max_gain_ratio > i):
                continue
            elif(max_gain_ratio < i):
                max_gain_ratio = i
                max_gain_ratio_index = temp
            else:
                rd = random.randint(0,1)
                if(rd==1):
                    max_gain_ratio = i
                    max_gain_ratio_index = temp
            temp += 1
        return max_gain_ratio_index
        
    else: # 基尼系数法
        Gini_index_list = []
        for i in dataset:
            Gini_index_list.append(Gini_index(i,y))
        Gini_min = 0
        Gini_min_index = 0
        for i in range(len(dataset)):
            if(Gini_min > Gini_index_list[i]):
                Gini_min = Gini_index_list[i]
                Gini_min_index = i
            elif(Gini_min == Gini_index_list[i]):
                rd = random.randint(0,1)
                if(rd==1):
                    Gini_min = Gini_index_list[i]
                    Gini_min_index = i
        return Gini_min_index

def Vote(y):
    classified = {} # 字典，key为dataset中的类别名称，value为类别对应的数量
    for i in y:
        if(i not in classified):
            classified[i] = 0
        classified[i] += 1
    max_number_classify = 0 # 初始化最大数量
    max_x = y[0]
    for x in classified:
        if(classified[x] > max_number_classify):
            max_number_classify = classified[x]
            max_x = x
        elif(classified[x] == max_number_classify): # 若类别数量相同，则随机选择
            rd = random.randint(0,1)
            if(rd==1):
                max_number_classify = classified[x]
                max_x = x
    return max_x


def Create_Decision_Tree(dataset, y,attribute, method = 'id3'):
    """
    构造决策树
    @dataset shape=[a,x] 全部待计算数据 a为属性个数，x为数据个数
    @y 训练数据划分类型结果 shape=[x,]
    @attribute 属性，shape=[a,]
    @method 选择最优划分属性方法，id3 为信息增益法，
            c4.5 为信息增益率法，cart为选择基尼系数法
    @return decision_tree 决策树
    """
    # 判断情况1：
    # 所有样例的类别一致，标记为叶节点，返回此类别
    temp = set(y)
    temp = list(temp)
    if(len(temp) == 1):
        return temp[0]
    
    # 判断为情况2：
    # 当前属性集为空或所有样本在所有属性上取值相同，无法划分，
    # 标记为叶节点，并设置类别为该节点所含样本最多的类别，返回
    if len(attribute) == 0:
        return Vote(y)
    flag = False
    length = []
    for l in dataset:
        temp = set(l)
        temp = list(temp)
        length.append(len(temp))
    length = set(length)
    length = list(length)
    if(len(length) == 1):
        flag = True     # 所有样本在所有属性上取值相同
    if(flag):
        return Vote(y)
    
    # 判断为情况3：
    # 选择最优属性划分
    index = Best_label(copy.deepcopy(dataset), y,method) # 最优划分属性下标
    data = copy.deepcopy(dataset[index]) # 选中的属性对应的行
    other_data = copy.deepcopy(dataset)
    other_data.pop(index) # 未选中的行  
    
    dic = attribute[index]  # attr 为字典,包含此属性的所有取值
    attr = []
    decision_tree = {} # 决策树
    label = ''  # 属性名
    for i in dic:
        attr=(dic[i])
        decision_tree[i] = {} # 字典中存字典
        label = i
    # 统计每个属性取值对应data中的个数
    number_attr = {}
    for i in attr:  # 初始化
        number_attr[i] = 0
    for i in data:
        number_attr[i]+=1
    # 子代的attribute剩余表
    son_attribute = copy.deepcopy(attribute)
    son_attribute.pop(index)
    
    for v in attr:
        # 当此属性值下为空，即data中此属性值的数量为0
        if(number_attr[v] == 0):
            decision_tree[label][v] = Vote(y)
            return decision_tree
        son_data = copy.deepcopy(other_data)
        son_y = []
        i = 0
        for j in range(len(data)):
            if(data[j] == v):
                son_y.append(y[j])
            else:
                for ls in son_data:
                    ls.pop(j-i)
                i+=1
        decision_tree[label][v] = Create_Decision_Tree(son_data,son_y, son_attribute, method)
    return decision_tree
    

def classify(Tree,attribute_name,attribute_value):
    """
    利用已经训练好的决策树对一个数据判断类型
    @Tree 训练好的决策树
    @attribute_name 属性名称
    @attribute_value 属性名称对应的取值
    @return classLabel  分类结果
    """
    classLabel = 0
    root = list(Tree.keys())[0]
    firstChildren = Tree[root]
    attr_index = attribute_name.index(root)
    for key in firstChildren.keys():
        if(attribute_value[attr_index] == key):
            if(type(firstChildren[key]) == type({})):
                classLabel = classify(firstChildren[key],attribute_name,attribute_value)
            else:
                classLabel = firstChildren[key]
    return classLabel

def dtClassify(decisionTree, attribute_name,attribute_value):
    """
    使用模型对一组数据分类
    @decisionTree 训练好的决策树
    @attribute_name 属性名称 shape = [a,]
    @attribute_value 属性名称对应的取值 shape = [a,x]
    @return classLabel shape=[a,] 结果list
    """
    classLabel = []
    for i in range(len(attribute_value[0])):
        attr_value = []
        attr_name = copy.deepcopy(attribute_name)
        for x in attribute_value:
            attr_value.append(x[i])
        classLabel.append(classify(decisionTree, attr_name, attr_value))
            
    return classLabel

def core(train_y,test_y):
    num = 0
    for i in range(len(train_y)):
        if(train_y[i] == test_y[i]):
            num+=1
    return num/len(train_y)



def StoreTree(Tree,filename):
    """
    存储决策树为json格式
    @Tree 训练好的决策树
    @filename 文件路径+名称
    """
    with open(filename,"w",encoding="utf-8") as f:
        json.dump(Tree,f,ensure_ascii=False)

def ReadTree(filename):
    """
    存储决策树为json格式
    @filename 文件路径+名称
    @return 返回训练好的决策树
    """
    with open(filename,'r',encoding="utf-8") as load_f:
        load_dict = json.load(load_f)
        print(load_dict)
    return load_dict
