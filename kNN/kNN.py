#!/usr/bin/env python
from numpy import *
import collections
import operator
import sys

def calEuclideanDist(inVector, dataSet):
	dataSetSize = dataSet.shape[0]
	diffMat = tile(inVector, (dataSetSize, 1)) - dataSet
	sqDiiffMat = diffMat ** 2
	sqDist = sqDiiffMat.sum(axis=1)
	return sqDist ** 0.5

def calManhattanDist(inVector, dataSet):
	dataSetSize = dataSet.shape[0]
	diffMat = tile(inVector, (dataSetSize, 1)) - dataSet
	absDiffMat = abs(diffMat)
	absDist = absDiffMat.sum(axis=1)
	return absDist

def calCosineMeasureDist(inVector, dataSet):
	sqDist = apply_along_axis(calCosineMeasure, 1, dataSet, inVector)
	return sqDist

def calCosineMeasure(x, y):
	ret = x.dot(y)
	return float(ret) / (sqrt(x.dot(x)) * sqrt(y.dot(y)))

def classify(testSet, dataSet, labels, k):
	if isinstance(testSet, (tuple, list)):
		for vector in testSet:
			classifyOneVec(vector, dataSet, labels, k)
	else:
		raise TypeError, 'invalid type of testSet in Function [classify]'

def classifyOneVec(inVector, dataSet, labels, k, distFunc):
	distances = distFunc(inVector, dataSet)
	sortedDistIndexList = distances.argsort()
	classCount = collections.defaultdict(int)
	for i in sortedDistIndexList[:k]:
		classCount[labels[i]] += 1
	sortedClassCount = sorted(
		classCount.iteritems(), key=operator.itemgetter(1), reverse=True
	)
	return sortedClassCount[0][0]

def autoNorm(dataSet):
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	ranges = maxVals - minVals
	normDataSet = zeros(dataSet.shape)
	m = dataSet.shape[0]
	normDataSet = dataSet - tile(minVals, (m, 1))
	normDataSet = normDataSet / tile(ranges, (m, 1))
	return normDataSet, ranges, minVals

def file2matrix(fname):
	fin = open(fname, 'r')
	rawDataLines = fin.readlines()
	fin.close()

	nLine = len(rawDataLines)
	rawData = rawDataLines[0]
	'calculate the num of attribute'
	nAttr = len(rawData.split()) - 1
	retMat = zeros((nLine, nAttr))
	retLabel = []
	for i,rawData in enumerate(rawDataLines):
		listFromData = rawData.strip().split()
		retMat[i, :] = listFromData[:-1]
		retLabel.append(int(listFromData[-1]))
	return retMat, retLabel

def kNNTest(dataMat, dataLabel, k, distFunc):
	testSetRatio = 0.1
	normMat, ranges, minVals = autoNorm(dataMat)
	m = dataMat.shape[0]
	nTestVec = int(m*testSetRatio)
	nError = 0
	for i in range(nTestVec):
		classifierResult = classifyOneVec(
			normMat[i, :], normMat[nTestVec:, :], dataLabel[nTestVec:], k, distFunc
		)
		if classifierResult!=dataLabel[i]: nError += 1
		print "the classifier's result: %d, real result: %d"\
					% (classifierResult, dataLabel[i])
	print 'the total error rate is: %f' % (float(nError)/float(nTestVec))


distFuncDist = {
	'Euclidean': calEuclideanDist,
	'ManhattanDist': calManhattanDist,
	'CosineMeasure': calCosineMeasureDist,
}

if __name__ == '__main__':
	fname = 'datingTestSet.txt'
	if len(sys.argv) > 1:
		fname = sys.argv[1]
	dataMat, dataLabel = file2matrix(fname)
	k = 3
	for distFuncName, distFunc in distFuncDist.iteritems():
		print '\nuse %s...' % (distFuncName)
		kNNTest(dataMat, dataLabel, k, distFunc)
