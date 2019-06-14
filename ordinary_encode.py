# -*-coding:utf-8-*-
import pandas as pd
import numpy as np
import math
from pandas.core.frame import DataFrame
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

train = pd.read_csv('smalltrain.csv',sep=',')
# print(train)
train = train.drop(['Row'],axis=1)
train = train.fillna(str(0))
train.iloc[:,0:] = OrdinalEncoder().fit_transform(train.iloc[:,0:])
# X = train[['Anon Student Id','Problem Name']]
# enc = OrdinalEncoder().fit(X)
# print(enc.categories_)
# print(enc.get_feature_names())
# print(len(enc.get_feature_names()))
print(train.head())
train.to_csv('smalltrain_oh.csv',index=0)
