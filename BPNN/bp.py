"""
Author: quxiangjun 20186471
Created on 2020.10.31 9:00
"""
import numpy as np
import copy
import json
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split


def sigmoid(x):
    """
    @desctribe: sigmoid function
    """
    f = 1.0 / (1+np.exp(-x))
    return f


def sigmoid_derivative(x):
    """
    @describe: sigmoid's derivative function
    """
    return sigmoid(x)*(1-sigmoid(x))


def OneHotEncoder(y):
    """
    @describe: oneHot编码
    @y: [N,] 每个数字代表1所在的位置
    @return oneHot_y [N,max_y]
    """
    max_y = 0  # (0~N)
    for i in y:
        max_y = max(max_y, i)
    oneHot_y = np.zeros((len(y), max_y+1))
    for i in range(len(y)):
        index = y[i]
        oneHot_y[i][index] = 1
    return oneHot_y


class bpNet():
    """
    @describe: bpNN
    """

    def __init__(self, input_num, hidden_num, output_num, learning_rate):
        """
        @describe: 初始化神经网络结构
        @input_num: 输入层神经元的个数
        @hidden_num: 单隐藏层神经元个数
        @output_num: 输出层神经元个数
        @learning_rate: 学习率
        """
        self.input_num = input_num  # 输入层神经元的个数
        self.hidden_num = hidden_num  # 隐藏层神经元个数
        self.output_num = output_num  # 输出层神经元个数
        # input_num行hidden_num列 # 输入层到隐藏层的系数
        self.W1 = np.random.rand(self.input_num, self.hidden_num)
        # 1行hidden_num列 输入层到隐藏层的常数
        self.b1 = np.random.rand(1, self.hidden_num)
        self.W2 = np.random.rand(
            self.hidden_num, self.output_num)  # hidden_num行output_num列 #隐藏层到输出层的系数
        # 1行output_num列 # 隐藏层到输出层的常数
        self.b2 = np.random.rand(1, self.output_num)
        self.learning_rate = learning_rate  # 学习率

    def forward(self, x):
        """
        @describe: 向前传播 Forward process of this simple network.
        @Input:
            x: np.array with shape [n,input_num] n 行数据，每行input_num个特征
        @return Y [N,output_num]
        """
        # 计算隐层的输入h
        # x [N,input_num] .dot W1 [input_num, hidden_num] = alfa [N,hidden_num]
        # alfa [N,hidden_num] 隐层神经元的输入 ： 第 N 个输入训练数据的 hidden_num 个隐层输入alfa
        self.alfa = x.dot(self.W1)

        # 计算隐层的输出
        # 输出b [N,hidden_num] = alfa [N,hidden_num] + self.b1 [1,hidden_num]隐藏层常数
        # b = sigmoid(b) 激活函数
        self.b = self.alfa + self.b1
        self.b = sigmoid(self.b)

        # 计算输出层的输入
        # b [N,hidden_num] .dot W2 [hidden_num, output_num] = beta [N,output_num]
        # beta [N,output_num] 输出层神经元的输入 ： 第 N 个输入训练数据的 output_num 个隐层输入beta
        self.beta = self.b.dot(self.W2)

        # 计算输出层输出
        # 输出Y [N,output_num] = beta [N,output_num] + self.b2 [1,output_num]隐藏层常数
        # Y = sigmoid(Y) 激活函数
        Y = self.beta + self.b2
        Y = sigmoid(Y)

        # 返回向前传播得到的结果Y [N,output_num] 第N个数据的输出output_num个结果
        return Y

    def Errors(self, y, train_result_y):
        """
        @describe: 计算误差
        @y: 训练数据集的label [N,output_num]
        @train_result_y: 前向传播的训练结果 [N,output_num] N组数据的结果Output_num个label
        @return error  N组数据集的误差
        """
        # 计算均方误差
        # error [N,output_num] = pow( (train_result_y [N,output_num] - y [1,output_num]) , 2)
        # error [N,] = 0.5 * sum(error.T)
        # error [N,] = sum(error)/len(error)
        error = pow((train_result_y - y), 2)
        error = sum(error.T) / len(error.T)
        if error.shape == (1,):
            error = error.reshape(1, 1)
        error = sum(error)/len(error)
        return error

    def grad(self, x, y, train_result_y):
        """
        @describe: 计算更新的梯度
        @x: 训练数据集 [N,input_num]
        @y: 训练数据集的label [N,output_num]
        @train_result_y: 前向传播的训练结果 [N,output_num] N组数据的结果Output_num个label
        """
        N = len(train_result_y)  # N组训练数据

        # 计算W2的迭代值
        #
        # G [N,output_num] = train_result_y  [N,output_num] * (1 - train_result_y ) * (y - train_result_y)
        #  for i in range(N): 遍历每一个数据集的误差
        #       self.grad_W2[output_num,hidden_num]  += self.learning_rate  * G[i].T [output_num,1] .dot( self.b[i] [1,hidden_num] 隐层的输出 )
        #  self.grad_W2 /= N 求所有的均值 [output_num,hidden_num]
        #  self.grad_W2.T [hidden_num,output_num]
        G = train_result_y * (1 - train_result_y) * (y - train_result_y)
        self.grad_W2 = np.zeros((self.output_num, self.hidden_num))
        for i in range(N):
            temp_G = np.array([G[i]])
            temp_b = np.array([self.b[i]])
            self.grad_W2 += self.learning_rate * (temp_G.T).dot(temp_b)
        self.grad_W2 /= N
        self.grad_W2 = self.grad_W2.T

        # 计算b2 迭代值
        #
        # for i in range(N): 遍历每一个数据集的误差
        #      self.grad_b2 [1,output_num] +=  -1 * self.learning_rate * G[i] [1,output_num]
        # self.grad_b2 /= N  [1,output_num]
        self.grad_b2 = np.zeros((1, self.output_num))
        for i in range(N):
            temp_G = np.array([G[i]])
            self.grad_b2 += -1 * self.learning_rate * temp_G
        self.grad_b2 /= N

        # 计算W1迭代值
        #
        # EH [N,hidden_num] = self.b [N,hidden_num] * (1-self.b)
        # temp [N,hidden_num] = G  [N,output_num] .dot(self.W2.T [output_num,hidden_num] )
        # EH [N,hidden_num] = EH * temp  [N,output_num]
        EH = self.b * (1-self.b)
        temp = G.dot(self.W2.T)
        EH = EH * temp
        # self.grad_W1 [input_num, hidden_num] = zeros((self.input_num, self.hidden_num))
        # for i in range(N):
        #     temp_EH [1,hidden_num]= np.array([EH[i]])
        #     temp_x [1,input_num] = np.array([x[i]])
        #     self.grad_W1  [input_num, hidden_num] += self.learning_rate * (temp_x.T  [input_num,1]).dot(temp_EH  [1,hidden_num])
        # self.grad_W1 /= N
        self.grad_W1 = np.zeros((self.input_num, self.hidden_num))
        for i in range(N):
            temp_EH = np.array([EH[i]])
            temp_x = np.array([x[i]])
            self.grad_W1 += self.learning_rate * (temp_x.T).dot(temp_EH)
        self.grad_W1 /= N

        # 计算b1的迭代值
        #
        # for i in range(N): 遍历每一个数据集的误差
        #     self.grad_b1 [1,hidden_num] += -1 * self.learning_rate * HE[i] [1,hidden_num]
        # self.grad_b1 /= N  [1,hidden_num]
        self.grad_b1 = np.zeros((1, self.hidden_num))
        for i in range(N):
            temp_EH = np.array([EH[i]])
            self.grad_b1 += -1 * self.learning_rate * temp_EH
        self.grad_b1 /= N

        return

    def update(self):
        """
        @describe: 更新网络参数
        """
        # 更新 W2
        # self.W2 [hidden_num, output_num] += self.grad_W2 [hidden_num,output_num]
        # self.W1 [input_num, hidden_num] += self.grad_W1 [input_num, hidden_num]
        # self.b1 [1, hidden_num] += self.grad_b1 [1,hidden_num]
        # self.b2 [1, output_num] += self.grad_b2 [1,output_num]
        self.W2 += self.grad_W2
        self.W1 += self.grad_W1
        self.b1 += self.grad_b1
        self.b2 += self.grad_b2

    def fit(self, x, y, times=1000, error_min=0.0000000000001):
        """
        @describe: 训练标准BP神经网络
        @x: 输入的训练数据集
        @y: 输入的训练数据集标记
        @times: 最大训练的次数 默认1000次
        @error_min: 最低的误差要求 默认最小迭代值0.0001
        """

        # 判断数据的规整性
        if(x.shape[1] != self.input_num or y.shape[1] != self.output_num):
            print("[ERROR] 数据规范性有问题 ")
            exit(1)

        error_old = 1  # 记忆均方误差

        while(times):
            times -= 1  # 迭代次数
            error_mean = 0  # 均方误差
            for i in range(len(x)):
                oneX = np.array([x[i]])
                oneY = np.array([y[i]])
                # 输出结果 [1,output_num] 输出类别特征有output_num个
                train_result_y = self.forward(oneX)  # shape[1,input_num]
                # grad
                self.grad(oneX, oneY, train_result_y)
                self.update()
                error = self.Errors(oneY, train_result_y)
                error_mean += error[0]  # 记忆error
            error_mean /= len(x)
            if times % 50 == 0:
                print("error_mean:", error_mean)
            if(error_old - error_mean < error_min) & (error_mean <= error_old):
                return
            error_old = error_mean

    def accumulated_fit(self, x, y, times=1000, error_min=0.0000000000001):
        """
        @describe: 训练累积误差BP神经网络
        @x: 输入的训练数据集
        @y: 输入的训练数据集标记
        @times: 最大训练的次数
        @error_min: 最低的误差要求
        """
        # 判断数据的规整性
        if(x.shape[1] != self.input_num or y.shape[1] != self.output_num):
            print("[ERROR] 数据规范性有问题 ")
            exit(1)

        error_old = 1  # 记忆均方误差

        while(times):
            times -= 1  # 迭代次数
            # 获得训练结果 [N,output_num] N组数据，每组数据有output_num个类别输出特征
            train_result_y = self.forward(x)  # shape[1,input_num]
            self.grad(x, y, train_result_y)
            self.update()
            error = self.Errors(y, train_result_y)
            if times % 50 == 0:
                print("error:", error)
            if(error_old - error < error_min) & (error <= error_old):
                return
            error_old = error

    def predict(self, x, y):
        """
        @describe: 预测结果
        @x: 测试数据集 [N,input_num]
        @y: 训练数据集的label [N,output_num]
        @return test_y,error 预测结果集合 [N,output_num]， 均方误差
        """
        result_y = self.forward(x)
        # print(result_y)
        error = self.Errors(y, result_y)
        # 取整test_y
        test_y = np.zeros(result_y.shape)
        for i in range(result_y.shape[0]):
            index_max = 0  # 找到最大的那一个
            for j in range(result_y.shape[1]):
                if(result_y[i][j] > result_y[i][index_max]):
                    index_max = j
            test_y[i][index_max] = 1
        correct_count = 0  # 正确的个数
        for i in range(len(y)):
            flag = True
            for j in range(len(y[0])):
                if(y[i][j] != test_y[i][j]):
                    flag = False
            if(flag):
                correct_count += 1
        return test_y, error, correct_count*1.0/len(y)

    def StoreBpNN(self, filename):
        """
        @describe: 存储神经网络到本地
        @filename: 存储文件名
        """
        dic = {}
        dic["input_num"] = self.input_num
        dic["hidden_num"] = self.hidden_num
        dic["output_num"] = self.output_num
        dic["learning_rate"] = self.learning_rate
        dic["W1"] = self.W1.tolist()
        dic["W2"] = self.W2.tolist()
        dic["b1"] = self.b1.tolist()
        dic["b2"] = self.b2.tolist()

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(dic, f, ensure_ascii=False)

    def LoadBpNN(filename):
        """
        @describe: 加载本地神经网络
        @filename: 加载的文件名
        @return bpNN 实例化好的dp网络
        """
        with open(filename, 'r', encoding="utf-8") as load_f:
            dic = json.load(load_f)
            # print(dic)
            bpNN = bpNet(
                dic["input_num"], dic["hidden_num"], dic["output_num"], dic["learning_rate"])
            bpNN.W1 = np.array(dic["W1"])
            bpNN.W2 = np.array(dic["W2"])
            bpNN.b1 = np.array(dic["b1"])
            bpNN.b2 = np.array(dic["b2"])
        return bpNN


def z_score(x):
    """
    @describe: 标准化数据集，z_score方法
    @x [N,input_num]
    @return diff_x : 归一化后的数据集
    """
    # mean_x [input_num,]= sum(x [N,input_num] ) / N
    # mean_x = mean_x.reshape(1,input_num)
    # diff_x [N,input_num] = x [N,input_num] - mean_x [1,input_num]
    # sigma2 [input_num,]= sum(diff_x*diff_x  [N,input_num])
    # sigma2 = sigma2.reshape(1,input_num)
    # for i in range(diff_x.shape[0]):
    #     for j in range(diff_x.shape[1]):
    #         diff_x[i][j] = diff_x[i][j] / sigma2[0][j]
    N = x.shape[0]
    input_num = x.shape[1]
    mean_x = sum(x) / N
    mean_x = mean_x.reshape(1, input_num)
    diff_x = x - mean_x
    sigma2 = sum(diff_x*diff_x) / N
    sigma2 = sigma2.reshape(1, input_num)
    for i in range(diff_x.shape[0]):
        for j in range(diff_x.shape[1]):
            diff_x[i][j] = diff_x[i][j] / sigma2[0][j]
    return diff_x


def Min_MaxNormalization(x):
    """
    @describe: 标准化数据集，z_score方法
    @x [N,input_num]
    @return diff_x : 归一化后的数据集
    """
    temp_x = x.T
    max_x = []
    min_x = []
    for item in temp_x:
        tmp = item.tolist()
        max_x.append(max(tmp))
        min_x.append(min(tmp))
    max_x = np.array(max_x)
    min_x = np.array(min_x)
    diff = max_x - min_x
    diff_x = (x - min_x)/diff
    return diff_x


if __name__ == "__main__":
    print("***********************************************************")
    print("******************* 开始训练iris数据集 *********************")
    print("***********************************************************")
    # 数据集、参数设置
    iris = datasets.load_iris()  # 加载莺尾花数据集
    x_train = iris.data
    y_train = iris.target

    x_train, x_test, y_train, y_test = train_test_split(
        x_train, y_train, test_size=0.25, random_state=0, stratify=y_train)
    y_train = OneHotEncoder(y_train)
    y_test = OneHotEncoder(y_test)
    input_num = len(x_train[0])
    hidden_num = 10
    output_num = len(y_train[0])
    learning_rate = 0.05

    print("******************标准BP iris 20186471*******************")
    bpNet_iris = bpNet(
        input_num, hidden_num, output_num, learning_rate)
    bpNet_iris.fit(x_train, y_train)
    test_result_y, error, correct_ratio = bpNet_iris.predict(x_test, y_test)
    print("测试结果")
    print(test_result_y)
    print("正确结果")
    print(y_test)
    print("输出层均方误差：", error)
    print("预测结果正确率：{:.2%}".format((correct_ratio)))
    bpNet_iris.StoreBpNN("BpNN_for_iris2.json")

    print("******************累积BP iris 20186471*******************")
    ac_bpNet_iris = bpNet(
        input_num, hidden_num, output_num, learning_rate)
    ac_bpNet_iris.accumulated_fit(x_train, y_train,9000)
    test_result_y2, error2, correct_ratio2 = ac_bpNet_iris.predict(x_test, y_test)
    print("测试结果")
    print(test_result_y2)
    print("正确结果")
    print(y_test)
    print("输出层均方误差：", error2)
    print("预测结果正确率：{:.2%}".format((correct_ratio2)))
    ac_bpNet_iris.StoreBpNN("ac_BpNN_for_iris2.json")

    # print("***********************************************************")
    # print("*********** 开始训练wine手写数字数据集 *************")
    # print("***********************************************************")
    # # 数据集、参数设置
    # wine = datasets.load_wine()  # 加载莺尾花数据集
    # x_train = wine.data
    # y_train = wine.target

    # x_train, x_test, y_train, y_test = train_test_split(
    #     x_train, y_train, test_size=0.25, random_state=0, stratify=y_train)
    # y_train = OneHotEncoder(y_train)
    # y_test = OneHotEncoder(y_test)

    # input_num = len(x_train[0])
    # hidden_num = 13
    # output_num = len(y_train[0])
    # learning_rate = 0.5

    # print("******************标准BP wine 20186471*******************")
    # x_train1 = Min_MaxNormalization(x_train)
    # x_test1 = Min_MaxNormalization(x_test)
    # bpNet_wine = bpNet(
    #     input_num, hidden_num, output_num, learning_rate)
    # bpNet_wine.fit(x_train1, y_train)
    # test_result_y, error, correct_ratio = bpNet_wine.predict(
    #     x_test1, y_test)
    # print("测试结果")
    # print(test_result_y)
    # print("正确结果")
    # print(y_test)
    # print("输出层均方误差：", error)
    # print("预测结果正确率：{:.2%}".format((correct_ratio)))
    # bpNet_wine.StoreBpNN("BpNN_for_wine2.json")

    # print("******************累积BP wine 20186471*******************")
    # x_train2 = z_score(x_train)
    # x_test2 = z_score(x_test)
    # ac_bpNet_wine = bpNet(
    #     input_num, hidden_num, output_num, learning_rate)
    # ac_bpNet_wine.accumulated_fit(x_train2, y_train, 3000)
    # test_result_y2, error2, correct_ratio2 = ac_bpNet_wine.predict(
    #     x_test2, y_test)
    # print("测试结果")
    # print(test_result_y2)
    # print("正确结果")
    # print(y_test)
    # print("输出层均方误差：", error2)
    # print("预测结果正确率：{:.2%}".format((correct_ratio2)))
    # ac_bpNet_wine.StoreBpNN("ac_BpNN_for_wine2.json")