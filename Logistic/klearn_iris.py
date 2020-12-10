from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
iris = datasets.load_iris()  # 加载莺尾花数据集
x_train = iris.data
y_train = iris.target

x_train,x_test,y_train,y_test = train_test_split(x_train,y_train,test_size = 0.25,random_state=0,stratify = y_train)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

regr = LogisticRegression()
regr.fit(x_train, y_train)

print('cofficients : %s,intercept %s'%(regr.coef_,regr.intercept_))
print('score : %.2f'%regr.score(x_test, y_test))
print(classification_report(y_test,regr.predict(x_test)))

print("predict value:",regr.predict(x_test))
print("test value:",y_test)


