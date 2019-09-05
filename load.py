

import urllib.request
import pandas as pd


def loadData():
    path = urllib.request.urlopen("https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data")
    data = pd.read_csv(path,sep = '\s+',names = ['SequenceName','mcg','gvh','alm','mit','erl','pox','vac','nuc','Class'])
    return data

#data = loadData()
#print(data)











