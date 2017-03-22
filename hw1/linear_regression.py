#coding:utf-8
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt

def loadData():
    rawData = pd.read_csv('./data/train.csv', na_values='NR', encoding='big5')
    rawData.ix[rawData[u'測項']=='RAINFALL',3:]=rawData.ix[rawData[u'測項']=='RAINFALL',3:].fillna(0)
    rawData.drop(rawData.columns[0:3], axis=1, inplace=True)
    return rawData

def sliceIntoTrunks(rawData):
    """split data by month"""
    trunk=[]
    rowOffset = 18
    rowIndex = 0
    indexes=range(rowOffset)
    for mon in range(12):
        monthTrunk = []
        for day in range(20):
            dataPerDay=rawData.ix[rowIndex:rowIndex+rowOffset-1]
            dataPerDay.index = indexes
            monthTrunk.append(dataPerDay)
            rowIndex += rowOffset
        trunk.append(pd.concat(monthTrunk, axis = 1))
    return trunk

def getData(dTrunk):
    i=0
    hours=9
    dataDict={}
    output=[]
    outputIndex=-9
    for data in dTrunk:
       for j in range(data.columns.size - hours + 1):
           vector=data.ix[:, j:j+hours].values.T.flatten()
           dataDict[i] = vector
           output.append(vector[outputIndex])
           i += 1

    output.pop(0)
    del dataDict[i-1]
    inputData=DataFrame(dataDict).T
    normalData=(inputData-inputData.mean())/inputData.std()
    normalData['b'] = 1
    inputData['b'] = 1
    return output, normalData, inputData

raw = loadData()
dataTrunk = sliceIntoTrunks(raw)
y,x, origX = getData(dataTrunk)

w = np.ones(x.columns.size)
base_rate=0.005
rate=base_rate
gradinetValue=1
gradientMax=1
count=0
e = y-x.dot(w)
prevErr = e.abs().mean() + 1
while gradinetValue > 0.000001 and base_rate > 0.00000001:
    e = y-x.dot(w)
    gradw = -2*e.dot(x)
    gradinetValue=np.sqrt(gradw.dot(gradw))
    rate= base_rate/gradinetValue if (gradinetValue > gradientMax) else base_rate
    w = w - rate*gradw
    err = e.abs().mean()
    if (count % 100 == 0): 
        descent = rate * gradinetValue
        loss=np.sqrt(e.dot(e)/e.index.size)
        print 'grad:',gradinetValue,'rate:', rate, 'descent:', descent,', loss:', loss,'err:', err, ', count:', count
    if (prevErr < err):
        base_rate /= 2
    else:
        base_rate *= 1.05
    prevErr = err
    count+=1
    #if (np.isinf(gradinetValue) or np.isinf(err)): 
    #    break

