import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from self_xigua import get_data
from Logistic import Logistic_train,Logistic_test
s = '011001101100110011001111111110001011011101011111011001111011111100011111010100000111100100111010110101110010010110100101111011100010111011010010111001010110101110110000010100011101010000110000000011110110011011101011111100'
data = [int(i) for i in s]
# print(data)
print(len(data))
x = []
for i in range(0,len(data)-200-1):
    x.append(data[i:200+i])
y = data[201:]
x = np.array(x)
x = x.T
y = np.array(y)
# regr = LogisticRegression()

# regr.fit(x, y)
beta = [[0] for i in range(200)]
beta [-1][0] = 1 
beta = np.array(beta)
print(beta)
beta = Logistic_train(x,y,beta)

# print('cofficients : %s,intercept %s'%(regr.coef_,regr.intercept_))
# print('score : %.2f'%regr.score(x, y))
# print(classification_report(y,regr.predict(x)))

test_x = data[-200:]
test_x = [test_x]
print(len(test_x))
test_x = np.array(test_x)
predict = Logistic_test(test_x,beta.T[0])
print("predict value:",predict)
# print("test value:",y)