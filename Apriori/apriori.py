#!/usr/bin/env python
from numpy import *

def loadDataSet():
	return [[1,3,4,6], [2,3,5,6], [1,2,3,5,6], [2,5,6]]

def createC1(dataSet):
	C1 = []
	for transaction in dataSet:
		for item in transaction:
			if not [item] in C1:
				C1.append([item])
	C1.sort()
	return map(frozenset, C1)

def scanD(D, Ck, minSupport):
	ssCnt = {}
	for tid in D:
		for can in Ck:
			if can.issubset(tid):
				if not ssCnt.has_key(can):
					ssCnt[can] = 1
				else:
					ssCnt[can] += 1
	numItems = float(len(D))
	retList = []
	supportData = {}
	for key in ssCnt:
		support = ssCnt[key] / numItems
		if support >= minSupport:
			retList.insert(0, key)
		supportData[key] = support
	return retList, supportData

def aprioriGen(Lk, k):
	retList = []
	lenLk = len(Lk)
	for i in range(lenLk):
		for j in range(i+1, lenLk):
			L1 = list(Lk[i])[:k-2]
			L2 = list(Lk[j])[:k-2]
			L1.sort()
			L2.sort()
			if L1 == L2:
				retList.append(Lk[i] | Lk[j])
	return retList

def apriori(dataSet, minSupport=0.5):
	C1 = createC1(dataSet)
	D = map(set, dataSet)
	L1, supportData = scanD(D, C1, minSupport)
	L = [L1]
	k = 2
	while len(L[k-2])>0:
		Ck = aprioriGen(L[k-2], k)
		print 'Ck =', Ck
		Lk, supK = scanD(D, Ck, minSupport)
		supportData.update(supK)
		# print supportData
		L.append(Lk)
		k += 1
	return L, supportData

def calcConf(freqSet, H, supportData, brl, minConf=0.7):
	prunedH = []
	for conseq in H:
		conf = supportData[freqSet] / supportData[freqSet-conseq]
		if conf>=minConf:
			print freqSet-conseq, '-->', conseq, 'conf:', conf
			brl.append((freqSet-conseq, conseq, conf))
			prunedH.append(conseq)
	return prunedH

def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
	m = len(H[0])
	print 'H =', H
	if len(freqSet) > m+1:
		Hmp1 = aprioriGen(H, m+1)
		print 'Hmp1(before calc) =', Hmp1
		Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
		print 'Hmp1(after calc) =', Hmp1
		if len(Hmp1) > 1:
			rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)


def generateRules(L, supportData, minConf=0.7):
	bigRuleList = []
	for i in range(1, len(L)):
		for freqSet in L[i]:
			print 'freqSet =', freqSet
			H1 = [frozenset([item]) for item in freqSet]
			if i>1:
				rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
			else:
				calcConf(freqSet, H1, supportData, bigRuleList, minConf)
	return bigRuleList


if __name__ == '__main__':
	dataSet = loadDataSet()
	# C1 = createC1(dataSet)
	# print 'C1 =', C1
	# D = map(set, dataSet)
	# print 'D =', D
	# L1, suppData0 = scanD(D, C1, 0.5)
	# print 'L1 =', L1
	L, supportData = apriori(dataSet)
	print 'L =', L
	rules = generateRules(L, supportData, 0.5)
	for i,rule in enumerate(rules):
		print '%d:rule = %s' % (i, str(rule))
	# for i,Li in enumerate(L):
	# 	print 'L[%d] = %s' % (i, str(Li))
	# print 'supportData =', supportData
