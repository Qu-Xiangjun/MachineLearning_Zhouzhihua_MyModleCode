from bp import *
from sklearn import datasets
from sklearn.model_selection import train_test_split

# 数据集、参数设置
iris = datasets.load_iris()  # 加载莺尾花数据集
x_train = iris.data
y_train = iris.target
x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, test_size=0.25, random_state=0, stratify=y_train)
y_train = OneHotEncoder(y_train)
y_test = OneHotEncoder(y_test)

print("******************标准BP iris 20186471*******************")
bpNet_iris = bpNet.LoadBpNN("BpNN_for_iris.json")
test_result_y, error,correct_ratio = bpNet_iris.predict(x_test, y_test)
# print("测试结果")
# print(test_result_y)
# print("正确结果")
# print(y_test)
print("输出层均方误差：", error)
print("预测结果正确率：{:.2%}".format((correct_ratio)))

print("******************累积BP iris 20186471*******************")
ac_bpNet_iris = bpNet.LoadBpNN("ac_BpNN_for_iris.json")
test_result_y2, error2,correct_ratio2 = ac_bpNet_iris.predict(x_test, y_test)
print("输出层均方误差：", error2)
print("预测结果正确率：{:.2%}".format((correct_ratio2)))

# 数据集、参数设置
wine = datasets.load_wine()  # 加载莺尾花数据集
x_train = wine.data
y_train = wine.target

x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, test_size=0.25, random_state=0, stratify=y_train)
y_train = OneHotEncoder(y_train)
y_test = OneHotEncoder(y_test)
x_test1 = Min_MaxNormalization(x_test)
x_test2 = z_score(x_test)

print("******************标准BP wine 20186471*******************")
BpNN_for_wine = bpNet.LoadBpNN("BpNN_for_wine.json")
test_result_y, error, correct_ratio = BpNN_for_wine.predict(
    x_test1, y_test)
print("输出层均方误差：", error)
print("预测结果正确率：{:.2%}".format((correct_ratio)))

print("******************累积BP wine 20186471*******************")
ac_BpNN_for_wine = bpNet.LoadBpNN("ac_BpNN_for_wine.json")
test_result_y2, error2,correct_ratio2 = ac_BpNN_for_wine.predict(x_test2, y_test)
print("输出层均方误差：", error2)
print("预测结果正确率：{:.2%}".format((correct_ratio2)))
