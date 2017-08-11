# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 11:29:53 2017

@author: cfd
"""
from os import listdir
from numpy import *
import operator 
#处理一个照片文件的方法
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect
#收集数据
def handwritingMatrix(dirname):
    hwLabels=[]
    trainingFileList = listdir(dirname)#(r'D:\Learning\DataSet\digits\trainingDigits'
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = fileStr.split('_')[0]
        hwLabels.append(int(classNumStr))
        trainingMat[i, :] = img2vector('%s/%s' % (dirname,fileNameStr))
    
    return trainingMat,hwLabels

def classifyhandwriting(handWritingData, DataSet, Labels, k):
    DataSetSize = DataSet.shape[0]
    DiffMat = tile(handWritingData, (DataSetSize, 1)) - DataSet
    sqDiffMat = DiffMat**2
    sqDistances = sqDiffMat**0.5
    distances = sqDistances.sum(axis=1)
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voterLabel = Labels[sortedDistIndicies[i]]
        classCount[voterLabel] = classCount.get(voterLabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
    
def handwritingClassTest():
    errorCount = 0.0
    testMat,TestLabels= handwritingMatrix(r'D:\Learning\DataSet\digits\testDigits')
    trainingMat, trainingLabels=handwritingMatrix(r'D:\Learning\DataSet\digits\trainingDigits')
    for i in range(testMat.shape[0]):
        Result = classifyhandwriting(testMat[i, :], trainingMat, trainingLabels, 3)
        print('the classifier came back with: %d, the real answer is: %d'\
              % (Result, TestLabels[i]))
        if(Result != TestLabels[i]):
            errorCount += 1
    print('\nthe total number of errors is: %d' % errorCount)
    print('\nthe total error rate is : %f' % (errorCount/float(testMat.shape[0])))
#print(handwritingMatrix())
handwritingClassTest()