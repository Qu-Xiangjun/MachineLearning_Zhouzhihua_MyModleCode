from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from self_xigua import get_data

x,y,other = get_data() # 由于数据太小，不做划分，封闭测试
x = x.T
# print(x.shape)
# print(y.shape)

regr = LogisticRegression()
regr.fit(x, y)

print('cofficients : %s,intercept %s'%(regr.coef_,regr.intercept_))
print('score : %.2f'%regr.score(x, y))
print(classification_report(y,regr.predict(x)))

print("predict value:",regr.predict(x))
print("test value:",y)