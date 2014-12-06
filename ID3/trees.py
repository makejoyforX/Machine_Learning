from math import log
import operator
import copy

def calcShannonEnt(dataSet):
	numEntries = len(dataSet)
	labelCounts = {}
	for featVec in dataSet:
		currentLabel = featVec[-1]
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1
	shannonEnt = 1
	for key in labelCounts:
		prob = float(labelCounts[key]) / numEntries
		shannonEnt -= prob * log(prob, 2)
	return shannonEnt
		
def createDataSet():
	dataSet = [ [1, 1, 'yes'],
				[1, 1, 'yes'],
				[1, 0, 'no' ],
				[0, 1, 'no' ],
				[0, 1, 'no' ]]
	labels = ['no surfacing', 'flippers']
	return dataSet, labels
	
def splitDataSet(dataSet, axis, value):
	retDataSet = []
	for featVec in dataSet:
		if featVec[axis] == value:
			reducedFeatVec = featVec[:axis]
			reducedFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reducedFeatVec)
	return retDataSet
	
def chooseBestFeatureToSplit(dataSet):
	numFeatures = len(dataSet[0]) - 1
	baseEntropy = calcShannonEnt(dataSet)
	bestInfoGain = 0.0
	bestFeature = -1
	for i in range(numFeatures):
		featList = [example[i] for example in dataSet]
		uniqueVals = set(featList)
		newEntropy = 0.0
		for value in uniqueVals:
			subDataSet = splitDataSet(dataSet, i, value)
			prob = len(subDataSet) / float(len(dataSet))
			newEntropy += prob * calcShannonEnt(subDataSet)
		infoGain = baseEntropy - newEntropy
		if infoGain > bestInfoGain:
			# print 'infoGain =', infoGain
			bestInfoGain = infoGain
			bestFeature = i
	return bestFeature
	
def majorityCnt(classList):
	classCounts = {}
	for vote in classList:
		if vote not in classCount.Keys():
			classCounts[vote] = 0
		classCounts[vote] += 1
	sortedClassCount = sorted(classCount.iteritems(),\
			key=operator.itemgetter(1), reverse = True)
	return sortedClassCount[0][0]
	
def createTree(dataSet, labels):
	classList = [example[-1] for example in dataSet]
	if classList.count(classList[0]) == len(classList):
		return classList[0]
	if len(dataSet[0]) == 1:
		return majorityCnt(classList)
	bestFeat = chooseBestFeatureToSplit(dataSet)
	bestFeatLabel = labels[bestFeat]
	myTree = {bestFeatLabel:{}}
	del(labels[bestFeat])
	featValues = [example[bestFeat] for example in dataSet]
	uniqueVals = set(featValues)
	for value in uniqueVals:
		subLabels = labels[:]
		myTree[bestFeatLabel][value] = createTree(splitDataSet(\
					dataSet, bestFeat, value), subLabels)
	return myTree
	
def classify(inputTree, featLabels, testVec):
	firstStr = inputTree.keys()[0]
	secondDict = inputTree[firstStr]
	featIndex = featLabels.index(firstStr)
	for key in secondDict.keys():
		if testVec[featIndex] == key:
			if type(secondDict[key]).__name__ == 'dict':
				classLabel = classify(secondDict[key], featLabels, testVec)
			else:
				classLabel = secondDict[key]
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
		
if __name__ == "__main__":
	myDat, labels = createDataSet()
	label = copy.deepcopy(labels)
	# print chooseBestFeatureToSplit(myDat)
	# print myDat
	print myDat
	myTree = createTree(myDat, labels)
	print myTree
	print label
	classTree = classify(myTree, label, [1])
	print classTree
	