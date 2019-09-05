import pandas as pd
from sklearn import svm
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from load import loadData

def omitOutliers():
    dataSet = loadData()
    dataUse = dataSet.loc[:,'mcg':'nuc']

    result_1 = svm.OneClassSVM(gamma='auto',nu = 0.1).fit(dataUse).predict(dataUse)
    result_2 = IsolationForest().fit(dataUse).predict(dataUse)
    result_3 = LocalOutlierFactor().fit_predict(dataUse)


    result_2 = pd.DataFrame(result_2,columns=['Outliers']) #use IsolationForest to drop our outliers

    dataSet = dataSet.join(result_2)
    dataSet = dataSet[dataSet['Outliers'] != -1]


    return(dataSet)

#data = omitOutliers()
#print(data)

