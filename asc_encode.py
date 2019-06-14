from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import sys
# wine = load_wine()
# wine.data.shape#(178,13)
# print(wine.data)
#如果wine是一张表，应该长这样：
# import pandas as pd
# pd.concat([pd.DataFrame(wine.data),pd.DataFrame(wine.target)],axis=1)
# print(wine.feature_names)
# print(wine.target_names)
import pandas as pd
import numpy as np
import math
from pandas.core.frame import DataFrame

def transform(word):
	res = ''
	for i in word:
		t = ord(i) % 10
		res += str(t)
	return res

if __name__=="__main__":
	filename = sys.argv[1]
	traindata = pd.read_csv(filename+'.csv',sep=',')
	# traindata.drop(['Step Start Time','First Transaction Time','Correct Transaction Time','Step End Time','Step Duration (sec)','Correct Step Duration (sec)','Error Step Duration (sec)','Incorrects','Hints','Corrects'],axis=1,inplace=True)
	cc = list(traindata.columns)

	final = []
	for row in traindata.iterrows():
		ans = []
		for i in row[1]:
			if type(i) == float and math.isnan(i):
				ans.append(-1)
			elif type(i) == int or type(i) == float:
				ans.append(i)
			else:
				res = transform(i)
				if len(res)>30:
					res = res[0:30]
				ans.append(float(res))
		final.append(ans)
	pd2csv = DataFrame(final)
	# print(cc)
	pd2csv.columns = cc
	# print(pd2csv.columns)
	outname = filename + '_trans.csv'
	pd2csv.to_csv(outname, index=0)
	print(pd2csv)
