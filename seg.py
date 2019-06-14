from sklearn import tree #导入需要的模块
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd

traindata = pd.read_csv('train.csv',sep='\t')
traindata.drop(['Step Start Time','First Transaction Time','Correct Transaction Time','Step End Time','Step Duration (sec)','Correct Step Duration (sec)','Error Step Duration (sec)','Incorrects','Hints','Corrects'],axis=1,inplace=True)
# print(len(traindata.columns))
tpd = traindata[0:2000]
tpd.to_csv('./smalltrain.csv',index=0)
# print(traindata)
tpd = traindata[2000:2200]
tpd.to_csv('./smalltest.csv',index=0)
