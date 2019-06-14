from time import time
import datetime
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, recall_score
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import math
import numpy as np

def rmse(l1,l2):
    res = 0
    for i in range(len(l1)):
        res += (l1[i] - l2[i])**2
    res = math.sqrt(res/len(l1))
    return res

times = time()

train = pd.read_csv('smalltrain_oh.csv',sep=',')
test = pd.read_csv('smalltest_oh.csv',sep=',')
train_data = train.drop(['Correct First Attempt'],axis=1)
test_data = test.drop(['Correct First Attempt'],axis=1)
train_target = train['Correct First Attempt']
test_target = test['Correct First Attempt']
c_range = [1]

recallall = []
aucall = []
scoreall = []
for c in c_range:
    svm = SVC(kernel='linear', gamma='auto',degree=1,C=c,cache_size=5000,class_weight='balanced').fit(train_data,train_target)
    res = svm.predict(test_data)
    score = svm.score(test_data,test_target)
    recall = recall_score(test_target,res)
    auc = roc_auc_score(test_target,svm.decision_function(test_data))
    recallall.append(recall)
    aucall.append(auc)
    scoreall.append(score)
    print("under C %f, testing accuracy is %f, recall is %f, auc is %f" %(c,score,recall,auc))
    # print(datetime.datetime.fromtimestamp(time()-times).strftime("&M:%S:"))
print(max(aucall),c_range[aucall.index(max(aucall))])
plt.figure()
plt.plot(c_range,recallall,c='red',label='recall')
plt.plot(c_range,aucall,c='black',label='auc')
plt.plot(c_range,scoreall,c='orange',label='accuracy')
plt.legend()
plt.show()
