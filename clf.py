from sklearn import tree #导入需要的模块
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
train = pd.read_csv('wholetrain_trans.csv',sep=',')
test = pd.read_csv('wholetest_trans.csv',sep=',')
train_data = train.drop(['Correct First Attempt','Row'],axis=1)
test_data = test.drop(['Correct First Attempt','Row'],axis=1)

train_target = train['Correct First Attempt']
test_target = test['Correct First Attempt']
clf = tree.DecisionTreeClassifier(max_depth=9,min_samples_leaf=10)     #实例化
# rfc = RandomForestClassifier(n_estimators=27)
# Xtrain, Xtest, Ytrain, Ytest = train_test_split(traindata, target, test_size=0.3)
# print("Xtrain")
# print(Xtrain)
# print("Ytrain")
# print(Ytrain)
# print("Xtest")
# print(Xtest)
# print("Ytest")
# print(Ytest)
clf = clf.fit(train_data,train_target)
# rfc = rfc.fit(train_data,train_target)
score_c = clf.score(test_data, test_target)
# score_r = rfc.score(test_data, test_target)
joblib.dump(clf, "clf.m")
# joblib.dump(rfc,"small_test_r.m")
print(score_c)
# print(score_r)
import matplotlib.pyplot as plt
test = []
for i in range(20):
    clf = tree.DecisionTreeClassifier(max_depth=9
                                    ,criterion="gini"
                                    # ,random_state=30
                                    # ,splitter="random"
                                    ,min_samples_leaf = 10
                                    ,min_samples_split = i+2
                                    )
    clf = clf.fit(train_data,train_target)
    score = clf.score(test_data, test_target)
    test.append(score)
plt.plot(range(2,22),test,color="red",label="max_depth")
plt.legend()
plt.show()
# test = []
# for i in range(50,300,10):
#     rfc = RandomForestClassifier(n_estimators=i,n_jobs=-1)
#     rfc = rfc.fit(train_data,train_target)
#     score_r = rfc.score(test_data, test_target)
#     test.append(score_r)
# plt.plot(range(50,300,10),test,color="red",label="max_depth")
# plt.legend()
# plt.show()


# clf = clf.fit(X_train,y_train) #用训练集数据训练模型
# result = clf.score(X_test,y_test) #导入测试集，从接口中调用需要的信息
