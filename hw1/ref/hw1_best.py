#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 22:33:56 2017

@author: minihair
"""

import csv 
import numpy as np
import sys
from math import isnan
from itertools import product


maxpower=2

Data = []
for i in range(18):
	Data.append([])

n_row = 0
text = open(str(sys.argv[1]), 'r', encoding='big5')
#text = open('train.csv', 'r', encoding='big5')
row = csv.reader(text , delimiter=",")
for r in row:
    if n_row != 0:
        for i in range(3,27):
            if r[i] == "NR":
                Data[(n_row-1)%18].append( float( 0 ) )
            elif r[i] == "-1":
                Data[(n_row-1)%18].append( float( 'nan' ) )
            elif float(r[i]) <1.0 and (n_row-1)%18 == 0: # weird temprature
                Data[(n_row-1)%18].append( float( 'nan' ) )
            else:             
                Data[(n_row-1)%18].append( float( r[i] ) )
    n_row =n_row+1
text.close()


for i in range (18): #interpolate if only one missing data
    for index, x in enumerate(Data[i]):
        if isnan(x) and index%480!=0 and index%480!=480-1:
            Data[i][index]=0.5*(Data[i][index-1]+Data[i][index+1])

Data = np.asarray(Data)

Data = np.vstack((Data, np.zeros((2, 12*20*24))))

# converting wind data from polar coordinate to cartesian, r, cos, sin. 
(Data[14], Data[15], Data[16], Data[17], Data[18], Data[19]) = (Data[16], np.cos(np.radians(Data[15])), np.sin(np.radians(Data[15])), Data[17], np.cos(np.radians(Data[14])), np.sin(np.radians(Data[14])))

for j in range(2,maxpower+1):
    for i in range(14):
        Data = np.vstack((Data, Data[i]**j)) #everything except wind up to second order
#    Data = np.vstack((Data, Data[i]**3)) #everything except wind up to third order
#==============================================================================
# 
# 
# 
# Data = np.vstack((Data, Data[9]**2)) #pm2.5 order 2
# Data = np.vstack((Data, Data[9]**3)) #pm2.5 order 3
# #Data = np.vstack((Data, Data[9]**4)) #pm2.5 order 4
# 
# Data = np.vstack((Data, Data[8]**2)) #pm10 order 2
# #Data = np.vstack((Data, Data[8]**3)) #pm10 order 3
# #Data = np.vstack((Data, Data[8]**4)) #pm10 order 4                
#==============================================================================
                
#Data = np.vstack((Data, Data[14]**2))#windspeed order 2
Data = np.vstack((Data, (Data[14]*Data[15])**1))#wind (rcos)**1
Data = np.vstack((Data, (Data[14]*Data[16])**1))#wind (rsin)**1
Data = np.vstack((Data, (Data[14]*Data[15])**2))#wind (rcos)**2
Data = np.vstack((Data, (Data[14]*Data[16])**2))#wind (rsin)**2
Data = np.vstack((Data, Data[14]*Data[14]*Data[15]*Data[16]))#wind r**2*cos*sin

#Data = np.vstack((Data, Data[17]**2))#windspeedHR order 2
Data = np.vstack((Data, (Data[17]*Data[18])**1))#windHR (rcos)**2
Data = np.vstack((Data, (Data[17]*Data[19])**1))#windHR (rsin)**2
Data = np.vstack((Data, (Data[17]*Data[18])**2))#windHR (rcos)**2
Data = np.vstack((Data, (Data[17]*Data[19])**2))#windHR (rsin)**2
Data = np.vstack((Data, Data[17]*Data[17]*Data[18]*Data[19]))#windHR r**2*cos*sin

Data[[0, 9], :]=Data[[9, 0], :]
#Data = np.delete(Data, (1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16, 17, 18, 19), axis = 0)                


train_x = []
train_y = []
deleted_rows = 0

N = 10 #number of days included in the train_x plus train_y

for i in range(12): #months
    for j in range(24*20-N+1): #number of possible consecutive data
        if isnan(Data[0][480*i+j+N-1]):
            deleted_rows += 1
            next
        else:
            train_y.append(Data[0][480*i+j+N-1])#the PM2.5 at time N
            train_x.append([1]) #offset b
            for t, s in product(range(Data.shape[0]), range(N-1)): #number of parameters * the first N-1 time points
                if np.isnan(Data[t][480*i+j+s]):
                    del train_x[-1]
                    del train_y[-1]
                    deleted_rows += 1
                    break
                else:
                    train_x[(24*20-N+1)*i+j-deleted_rows].append(Data[t][480*i+j+s])
                    
                    
train_x = np.asarray(train_x)
train_y = np.asarray(train_y)  
                    









testD = []
for i in range(18):
	testD.append([])

n_row = 0
text = open(str(sys.argv[2]), 'r', encoding='big5')
#text = open('test_X.csv', 'r', encoding='big5')
row = csv.reader(text , delimiter=",")
for r in row:
    for i in range(11-N+1,11):
        if r[i] == "NR":
            testD[(n_row)%18].append( float( 0 ) )
        elif r[i] == "-1":
            testD[(n_row)%18].append( float( 'nan' ) )
        else:            
            testD[(n_row)%18].append( float( r[i] ) )
    n_row =n_row+1
text.close()

for i in range (18): #interpolate if only one missing data
    for index, x in enumerate(testD[i]):
        if isnan(x) and index%9!=0 and index%9!=9-1:
            testD[i][index]=0.5*(testD[i][index-1]+testD[i][index+1])
        if isnan(x) and index%9==0:
            testD[i][index]=testD[i][index+1]

testD = np.asarray(testD)

# converting wind data from polar coordinate to cartesian, r, cos, sin. 
testD = np.vstack((testD, np.zeros((2, 9*240))))
(testD[14], testD[15], testD[16], testD[17], testD[18], testD[19]) = (testD[16], np.cos(np.radians(testD[15])), np.sin(np.radians(testD[15])), testD[17], np.cos(np.radians(testD[14])), np.sin(np.radians(testD[14])))

for j in range(2,maxpower+1):
    for i in range(14):
        testD = np.vstack((testD, testD[i]**j)) #everything except wind up to second order
#    testD = np.vstack((testD, testD[i]**3)) #everything except wind up to third order
#==============================================================================
# testD = np.vstack((testD, testD[9]**2)) #pm2.5 order 2
# testD = np.vstack((testD, testD[9]**3)) #pm2.5 order 3
# #testD = np.vstack((testD, testD[9]**4)) #pm2.5 order 4
# 
# testD = np.vstack((testD, testD[8]**2)) #pm10 order 2
# #testD = np.vstack((testD, testD[8]**3)) #pm10 order 3
# #testD = np.vstack((testD, testD[8]**4)) #pm10 order 4    
#==============================================================================

#testD = np.vstack((testD, testD[14]**2))#windspeed order 2
testD = np.vstack((testD, (testD[14]*testD[15])**1))#wind (rcos)**1
testD = np.vstack((testD, (testD[14]*testD[16])**1))#wind (rsin)**1
testD = np.vstack((testD, (testD[14]*testD[15])**2))#wind (rcos)**2
testD = np.vstack((testD, (testD[14]*testD[16])**2))#wind (rsin)**2
testD = np.vstack((testD, testD[14]*testD[14]*testD[15]*testD[16]))#wind r**2*cos*sin

#testD = np.vstack((testD, testD[17]**2))#windspeedHR order 2
testD = np.vstack((testD, (testD[17]*testD[18])**1))#windHR (rcos)**1
testD = np.vstack((testD, (testD[17]*testD[19])**1))#windHR (rsin)**1
testD = np.vstack((testD, (testD[17]*testD[18])**2))#windHR (rcos)**2
testD = np.vstack((testD, (testD[17]*testD[19])**2))#windHR (rsin)**2
testD = np.vstack((testD, testD[17]*testD[17]*testD[18]*testD[19]))#windHR r**2*cos*sin

testD[[0, 9], :]=testD[[9, 0], :]
#testD = np.delete(testD, (1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16, 17, 18, 19), axis = 0)   

test=np.ones((240, (N-1)*testD.shape[0]+1))
for i in range(240):
    test[i, 1:(N-1)*testD.shape[0]+1]= np.ndarray.flatten(testD[:, i*(N-1):(i+1)*(N-1)])










feature_m = np.zeros(train_x.shape[1])
feature_s = np.ones(train_x.shape[1])

#for i in range(1, train_x.shape[1]): #omitting the first column. 
#    feature_m[i] = np.mean(train_x[:,i])
#    feature_s[i] = np.std(train_x[:,i])

n_train_x = (train_x-feature_m)/feature_s


#for subsize in range(100,99, -100):
subsize = 5000
#for i in range(9):    

for i in range(3000, 3000,200):
    
    
    rmse=[]
#    M=100
    l=i
    np.random.seed(10)
#    n=0

    while True:
        permutation = np.random.permutation(train_x.shape[0])
#        print(permutation)
        perm_x = train_x[permutation, :]
        perm_y = train_y[permutation]
        
        subtrain_x = perm_x[:subsize, :]
        subvalid_x = perm_x[subsize:, :]
        subtrain_y = perm_y[:subsize]
        subvalid_y = perm_y[subsize:]    
        
        subfeature_m = np.zeros(subtrain_x.shape[1])
        subfeature_s = np.ones(subtrain_x.shape[1])

#        for i in range(1, subtrain_x.shape[1]): #omitting the first column. 
#            subfeature_m[i] = np.mean(subtrain_x[:,i])
#            subfeature_s[i] = np.std(subtrain_x[:,i])

        n_subtrain_x = (subtrain_x-subfeature_m)/subfeature_s
        n_subvalid_x = (subvalid_x-subfeature_m)/subfeature_s
        
        
        lI_0 = l*np.eye(subtrain_x.shape[1])
        lI_0[0,0] = 0
    #    subw = np.dot(np.linalg.pinv(subtrain_x), subtrain_y)
        subw = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(n_subtrain_x),n_subtrain_x)+lI_0),np.transpose(n_subtrain_x)),subtrain_y)
        subresult = np.dot(n_subvalid_x, subw)
        #print(subresult)
#        print(np.sqrt(np.sum((subresult-subvalid_y)**2)/(5512-subsize)))
        rmse.append(np.sqrt(np.sum((subresult-subvalid_y)**2)/(5512-subsize)))
#        n +=1
#        print(n)
#        print(len(rmse))
        if (np.std(rmse)/np.sqrt(len(rmse)) < 0.05 and len(rmse)>10) or len(rmse)==200:
            break
    
    print(l, np.mean(rmse), np.std(rmse)/np.sqrt(len(rmse)), len(rmse))
    #print(rmse)





lI_0 = 2500*np.eye(train_x.shape[1])







#w = np.dot(np.linalg.pinv(train_x), train_y)
w = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(n_train_x), n_train_x) + lI_0), np.transpose(n_train_x))    , train_y)
    
n_test = (test-feature_m)/feature_s

result = np.dot(n_test, w)
for i in range(240):
    if result[i] <0:
        result[i] = 0
              
              
              
              
              
    
text_file = open(str(sys.argv[3]), "w")
#text_file = open('out.csv', "w")
text_file.write("id,value\n")

for i in range(240):
   text_file.write("id_%i,%f\n" % (i, result[i]) )
text_file.close()



