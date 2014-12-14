#!/usr/bin/env python
from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt

def createDataSet():
	group = array(
		[[1.0, 1.1], [1.0, 1.0], [0., 0.], [0., 0.1]]
	)
	labels = ['A', 'A', 'B', 'B']
	return group, labels

def classify0(inX, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]
	diffMat = tile(inX, (dataSetSize, 1)) - dataSet
	sqDiffMat = diffMat**2
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances**0.5
	sortedDistIndices = distances.argsort()
	classCount = {}
	for i in range(k):
		voteIlabel = labels[sortedDistIndices[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
	sortedClassCount = sorted(
		classCount.iteritems(), key=operator.itemgetter(1), reverse=True
	)
	return sortedClassCount[0][0]

def file2matrix(filename):
	fr = open(filename)
	arrayOfLines = fr.readlines()
	numberOfLines = len(arrayOfLines)
	returnMat = zeros((numberOfLines, 3))
	classLabelVector = []
	classLabels = {}
	index = 0
	for line in arrayOfLines:
		line = line.strip()
		listFromLine = line.split('\t')
		returnMat[index, :] = listFromLine[0:3]
		if listFromLine[-1] not in classLabels:
			classLabels[listFromLine[-1]] = len(classLabels)
		classLabelVector.append(classLabels[listFromLine[-1]])
		index += 1
	return returnMat, classLabelVector
	
def autoNorm(dataSet):
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	ranges = maxVals - minVals
	normDataSet = zeros(dataSet.shape)
	m = dataSet.shape[0]
	normDataSet = dataSet - tile(minVals, (m, 1))
	normDataSet = normDataSet / tile(ranges, (m, 1))
	return normDataSet, ranges, minVals

def showMatrix(matrix):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(
		matrix[:,1], matrix[:,2], 15.0*array(datingLabels), 15.0*array(datingLabels) 
	)
	plt.show()

def classfiyPerson():
	resultList = ['not at all', 'in small doses', 'in large doses']
	percentTats = float(raw_input("percentage of time spent playing video games?"))
	ffMiles = float(raw_input("frequent filer meils earned per year?"))
	iceCream = float(raw_input('liters of ice cream consumed per year?'))
	datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat)
	inArr = array([ffMiles, percentTats, iceCream])
	classifierResult = classify0(
		(inArr-minVals)/ranges, normMat, datingLabels, 3
	)
	print 'You will probably like this person: ', resultList[classifierResult-1]

if __name__ == "__main__":
	#group, labels = createDataSet()
	#print classify0([0., 0.], group, labels, 3)
	#datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
	# print datingDataMat[:,1]
	#showMatrix(datingDataMat)
	classfiyPerson()
	