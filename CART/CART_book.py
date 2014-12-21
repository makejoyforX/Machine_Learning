#!/usr/bin/env python

from numpy import *

def loadDataSet(fileName):
	dataMat = []
	fr = open(fileName)
	for line in fr.readlines():
		curLine = line.strip().split('\t')
		fltLine = map(float, curLine)
		dataMat.append(fltLine)
	fr.close()
	return dataMat

def binSplitDataSet(dataSet, feature, value):
	# x = nonzero(dataSet[:, feature] <= value)
	# print 'nonzero(dataSet[:, feature] <= value) =\n\t', x
	# print 'dataSet[x[0], :] =\n\t', dataSet[x[0], :]
	mat0 = dataSet[nonzero(dataSet[:, feature] >  value)[0], :][0]
	mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :][0]
	return mat0, mat1

def regLeaf(dataSet):
	return mean(dataSet[:,-1])

def regErr(dataSet):
	return var(dataSet[:,-1]) * shape(dataSet)[0]

def chooseBestSpilt(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
	tolS = ops[0]
	tolN = ops[1]
	if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
		return None, leafType(dataSet)
	m, n = shape(dataSet)
	S = errType(dataSet)
	bestS = inf; bestIndex = 0; bestValue = 0
	for featIndex in range(n-1):					# dataSet[n-1] is class
		for splitVal in set(dataSet[:,featIndex]):
			mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
			if shape(mat0)[0] < tolN or shape(mat1)[0] < tolN:
					continue
			newS = errType(mat0) + errType(mat1)
			if newS < bestS:
				bestIndex = featIndex
				bestValue = splitVal
				bestS = newS
	if S-bestS < tolS:
		return None, leafType(dataSet)
	mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
	if shape(mat0)[0] < tolN or shape(mat1)[0]<tolN:
		return None, leafType(dataSet)
	return bestIndex, bestValue

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
	feat, val = chooseBestSpilt(dataSet, leafType, errType, ops)
	if feat == None:
		return val
	retTree = {}
	retTree['spInd'] = feat
	retTree['spVal'] = val
	lSet, rSet = binSplitDataSet(dataSet, feat, val)
	retTree['left'] = createTree(lSet, leafType, errType, ops)
	retTree['right'] = createTree(rSet, leafType, errType, ops)
	return retTree

def isTree(obj):
	return type(obj).__name__ == "dict"

def getMean(tree):
	if isTree(tree['right']):
		tree['right'] = getMean(tree['right'])
	if isTree(tree['left']):
		tree['left'] = getMean(tree['left'])
	return (tree['left']+tree['right'])/2.0

def prune(tree, testData):
	if shape(testData)[0] == 0:
		return getMean(tree)
	if isTree(tree['right']) or isTree(tree['left']):
		lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
	if isTree(tree['left']):
		tree['left'] = prune(tree['left'], lSet)
	if isTree(tree['right']):
		tree['right'] = prune(tree['right'], rSet)
	if not isTree(tree['left']) and not isTree(tree['right']):
		lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
		errorNoMerge = sum(power(lSet[:,-1] - tree['left'], 2)) +\
					   sum(power(lSet[:,-1] - tree['right'], 2))
		treeMean = (tree['left'] + tree['right']) / 2.0
		errorMerge = sum(power(testData[:,-1] - treeMean, 2))
		if errorMerge < errorNoMerge:
			print 'merging'
			return treeMean
		else:
			return tree
	else:
		return tree

if __name__ == '__main__':
	# testMat = mat(eye(4))
	# print testMat
	# mat0, mat1 = binSplitDataSet(testMat, 1, 0.5)
	# print mat0
	# print mat1
	# myDat = loadDataSet('ex0.txt')
	# myMat = mat(myDat)
	# tree = createTree(myMat, regLeaf, regErr)
	# for key,val in tree.iteritems():
	# 	print key,'=>',val
	myDat2 = loadDataSet('ex2.txt')
	myMat2 = mat(myDat2)
	myTree = createTree(myMat2, ops=(0,1))
	myDataTest = loadDataSet('ex2test.txt')
	myMat2Test = mat(myDataTest)
	pruneTree = prune(myTree, myMat2Test)
	print pruneTree
