from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import graphviz

traindata = pd.read_csv('finaltest.csv',sep=',')
traindata.drop(['Correct First Attempt','Row'],axis=1,inplace=True)
clf = joblib.load("clf.m")
feature_name = list(traindata.columns)
# print(len(feature_name))
# dot_data = tree.export_graphviz(clf
#                                # ,out_file = None
#                                ,feature_names= feature_name
#                                ,filled=True
#                                ,rounded=True
#                                )
#
# graph = graphviz.Source(dot_data)
# graph.render("Tree")
# graph.view()
print([*zip(feature_name,clf.feature_importances_)])
