from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
import math

def rmse(l1,l2):
    sum = 0
    for i in range(len(l1)):
        sum += (l1[i] - l2[i])**2
    res = math.sqrt(sum/len(l1))
    return (sum,res)

rfc = joblib.load("rfc.m")
test = pd.read_csv('finaltest_oh.csv',sep=',')
test_data = test.drop(['Correct First Attempt'],axis=1)
test_target = test['Correct First Attempt']

score_r = rfc.score(test_data, test_target)
y = rfc.predict(test_data)
print(rmse(y,test_target))
print(score_r)
