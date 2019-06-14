from sklearn import tree #导入需要的模块
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import math

def rmse(l1,l2):
    res = 0
    for i in range(len(l1)):
        res += (l1[i] - l2[i])**2
    res = math.sqrt(res/len(l1))
    return res


train = pd.read_csv('wholetrain_trans.csv',sep=',')
test = pd.read_csv('wholetest_trans.csv',sep=',')
train_data = train.drop(['Correct First Attempt'],axis=1)
test_data = test.drop(['Correct First Attempt'],axis=1)
train_target = train['Correct First Attempt']
test_target = test['Correct First Attempt']
rgr = DecisionTreeRegressor(criterion='friedman_mse')
rgr = rgr.fit(train_data,train_target)
joblib.dump(rgr, "rgr.m")
y = rgr.predict(test_data)
print(rmse(y,test_target))
# print([y[i]-test_target[i] for i in range(len(y))])
