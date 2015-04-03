
class treeNode:
	def __init__(self, nameValue, numOccur, parentNode):
		self.name = nameValue
		self.count = numOccur
		self.nodeLink = None
		self.parent = parentNode
		self.children = {}
		
	def inc(self, numOccur):
		self.count += numOccur
		
	def disp(self, ind=1):
		print '  '*ind, self.name, ' ', self.count
		for child in self.children.values():
			# print child.name
			child.disp(ind+1)

			
def createTree(dataSet, minSup=1):
	headerTable = {}
	for trans in dataSet:
		for item in trans:
			headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
	for k in headerTable.keys():
		if headerTable[k] < minSup:
			del(headerTable[k])
	freqItemSet = set(headerTable.keys())
	if len(freqItemSet)==0:
		return None, None
	for k in headerTable:
		headerTable[k] = [headerTable[k], None]
	retTree = treeNode('Null', 1, None)
	for tranSet, count in dataSet.items():
		localD = {}
		for item in tranSet:
			if item in freqItemSet:
				localD[item] = headerTable[item][0]
		if len(localD) > 0:
			orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p:p[1], reverse=True)]
			# print 'orderedItems =', orderedItems
			updateTree(orderedItems, retTree, headerTable, count)
	return retTree, headerTable
	
	
def updateTree(items, inTree, headerTable, count):
	# print 'items =', items
	if items[0] in inTree.children:
		inTree.children[items[0]].inc(count)
	else:
		inTree.children[items[0]] = treeNode(items[0], count, inTree)
		if headerTable[items[0]][1] == None:
			headerTable[items[0]][1] = inTree.children[items[0]]
		else:
			updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
	if len(items)>1:
		updateTree(items[1:], inTree.children[items[0]], headerTable, count)
			

def updateHeader(nodeToTest, targetNode):
	while (nodeToTest.nodeLink != None):
		nodeToTest = nodeToTest.nodeLink
	nodeToTest.nodeLink = targetNode
	
	
def loadSimpData():
	simpDat = [
		['r', 'z', 'h', 'j', 'p'],
		['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
		['z'],
		['r', 'x', 'n', 'o', 's'],
		['y', 'r', 'x', 'z', 'q', 't', 'p'],
		['y', 'z', 'x', 'e', 'q', 's', 't', 'm'],
	]
	return simpDat
	
	
def createInitSet(dataSet):
	retDict = {}
	for trans in dataSet:
		retDict[frozenset(trans)] = 1
	return retDict
	
	
def ascendTree(leafNode, prefixPath):
	if leafNode.parent != None:
		prefixPath.append(leafNode.name)
		ascendTree(leafNode.parent, prefixPath)
		

def findPrefixPath(basePat, treeNode):
	condPats = {}
	while treeNode != None:
		prefixPath = []
		ascendTree(treeNode, prefixPath)
		if len(prefixPath) > 1:
			condPats[frozenset(prefixPath[1:])] = treeNode.count
		treeNode = treeNode.nodeLink
	return condPats

	
def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
	bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p:p[1])]
	for basePat in bigL:
		newFreqSet = preFix.copy()
		newFreqSet.add(basePat)
		freqItemList.append(newFreqSet)
		condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
		# print 'condPattBases:'
		# print condPattBases
		myCondTree, myHead = createTree(condPattBases, minSup)
		if myHead != None:
			# print 'conditional tree for: ',newFreqSet
			# myCondTree.disp(1) 
			mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)
			
if __name__ == '__main__':
	# rootNode = treeNode('pyramid', 9, None)
	# rootNode.children['eye'] = treeNode('eye', 13, None)
	# rootNode.disp()
	# rootNode.children['phoenix'] = treeNode('phoenix', 3, None)
	# rootNode.disp()
	# simpDat = loadSimpData()
	# print simpDat
	# initSet = createInitSet(simpDat)
	# print initSet
	# myFPtree, myHeaderTable = createTree(initSet, 3)
	# myFPtree.disp(1)
	# print myHeaderTable
	# print findPrefixPath('x', myHeaderTable['x'][1])
	# print findPrefixPath('z', myHeaderTable['z'][1])
	# print findPrefixPath('r', myHeaderTable['r'][1])
	# simpDat = loadSimpData()
	# initSet = createInitSet(simpDat)
	# myFPtree, myHeaderTable = createTree(initSet, 3)
	# freqItems = []
	# mineTree(myFPtree, myHeaderTable, 3, set([]), freqItems)
	# print 'freqItems :'
	# print freqItems
	parsedDat = [line.split() for line in open('kosarak.txt').readlines()]
	initSet = createInitSet(parsedDat)
	myFreqList = []
	myFPtree, myHeaderTable = createTree(initSet, 10**5)
	mineTree(myFPtree, myHeaderTable, 10**5, set([]), myFreqList)
	print 'len(myFreqList) =', len(myFreqList)
	print 'myFreqList =', myFreqList
	