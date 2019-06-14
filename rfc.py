from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
import math

def rmse(l1,l2):
    res = 0
    for i in range(len(l1)):
        res += (l1[i] - l2[i])**2
    res = math.sqrt(res/len(l1))
    return res

train = pd.read_csv('wholetrain_trans.csv',sep=',')
test = pd.read_csv('wholetest_trans.csv',sep=',')
train_data = train.drop(['Row','Correct First Attempt'],axis=1)
test_data = test.drop(['Row','Correct First Attempt'],axis=1)
train_target = train['Correct First Attempt']
test_target = test['Correct First Attempt']
# clf = tree.DecisionTreeClassifier(max_depth=6,min_samples_leaf=10,min_samples_split=10)
rfc = RandomForestClassifier(n_estimators=300,max_depth=9,min_samples_leaf=10)
rfc = rfc.fit(train_data,train_target)
score_r = rfc.score(test_data, test_target)
joblib.dump(rfc,"rfc.m")
y = rfc.predict(test_data)
print(rmse(y,test_target))
print(score_r)

# import matplotlib.pyplot as plt
# test = []
# begin = 1
# end = 21
# step = 1
# for i in range(begin,end,step):
#     rfc = RandomForestClassifier(n_estimators=100,n_jobs=-1,max_depth=9,min_samples_leaf=i)
#     rfc = rfc.fit(train_data,train_target)
#     score_r = rfc.score(test_data, test_target)
#     test.append(score_r)
# plt.plot(range(begin,end),test,color="red",label="max_depth")
# plt.legend()
# plt.show()
