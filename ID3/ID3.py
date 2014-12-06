from __future__ import division
from math import *
from collections import defaultdict
import operator
import treePlotter

def calEntropy(dataSet):
	labelCntDict = defaultdict(int)
	for vector in dataSet:
		labelCntDict[vector[-1]] += 1
	nEntry = len(dataSet)*1.0
	entropy = 0.0
	for key,cnt in labelCntDict.iteritems():
		pi = cnt / nEntry
		entropy -= pi * log(pi, 2)
	return entropy
	
def splitDataSet(dataSet, i, val):
	retDataSet = []
	for vector in dataSet:
		if vector[i] == val:
			retDataSet.append(vector[:i] + vector[i+1:])
	return retDataSet
		
def Get_bestAttr(dataSet):
	nAttr = len(dataSet[0]) - 1;
	'caluclate the base Entropy'
	baseEntropy = calEntropy(dataSet)
	bestInfoGain = 0.0
	bestAttr = -1
	print 'nAttr =', nAttr
	print dataSet
	for i in range(nAttr):
		valList = [vector[i] for vector in dataSet]
		valSet = set(valList)
		'calculate the gain of entropy'
		newEntropy = 0.0
		for val in valSet:
			subDataSet = splitDataSet(dataSet, i, val)
			prob = len(subDataSet) / len(dataSet)
			newEntropy += prob * calEntropy(subDataSet)
		infoGain = baseEntropy - newEntropy
		if infoGain > bestInfoGain:
			bestInfoGain = infoGain
			bestAttr = i
	return bestAttr
	
def majorityCnt(classList):
	classDict = defaultdict(int)
	for item in classList:
		classDict[item] += 1
	sortedClassList = sorted(
		classDict.iteritems(), key=operator.itemgetter(1)
	)
	return sortedClassList[-1][0]
	
def createTree(dataSet, Labels):
	classList = [vector[-1] for vector in dataSet]
	if classList.count(classList[0]) == len(classList):
		return classList[0]
	if len(dataSet[0]) == 1:
		'no other attr left'
		return majorityCnt(classList)
	bestAttr = Get_bestAttr(dataSet)
	bestAttrLabel = Labels[bestAttr]
	myTree = {bestAttrLabel:dict()}
	valSet = set([vector[bestAttr] for vector in dataSet])
	for val in valSet:
		subLabels = Labels[:bestAttr] + Labels[bestAttr+1:]
		myTree[bestAttrLabel][val] = createTree(
			splitDataSet(dataSet, bestAttr, val), subLabels
		)
	return myTree
	
def classify(inputTree, attrLabels, testVec):
	firstStr = inputTree.keys()[0]
	secondDict = inputTree[firstStr]
	attrIndex = attrLabels.index(firstStr)
	for key,val in secondDict.iteritems():
		if testVec[attrIndex] == key:
			if isinstance(val, dict):
				classLabel = classify(val, attrLabels, testVec)
			else:
				classLabel = val
	return classLabel
	
def storeTree(inputTree, filename):
	import pickle
	fw = open(filename, "w")
	pickle.dump(inputTree, fw)
	fw.close()
	
def grabTree(filename):
	import pickle
	fr = open(filename, "r")
	return pickle.load(fr)
	
def readLenses():
	attr_fname = 'lenses.attr'
	data_fname = 'lenses.data'
	'read the data file '
	attrDict = {}
	Labels = []
	'open the attr file'
	fin = open(attr_fname, 'r')
	i = 0
	for line in fin.readlines():
		if len(line)==0:
			continue
		wList = line.split()
		Labels.append(wList[0])
		attrDict[i] = wList[1:]
		i += 1
	fin.close()
	dataSet = []
	fin = open(data_fname, 'r')
	for line in fin.readlines():
		wList = line.split()
		vector = [attrDict[i][int(j)-1] for i,j in enumerate(wList[1:])]
		dataSet.append(vector)
	fin.close()
	return dataSet, Labels[:-1]
	
if __name__ == "__main__":
	dataSet, Labels = readLenses()
	print dataSet, Labels
	myTree = createTree(dataSet, Labels)
	print myTree
	treePlotter.createPlot(myTree)
	