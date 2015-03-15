from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName):
	dataMat = []
	fr = open(fileName)
	for line in fr.readlines():
		curLine = line.strip().split('\t')
		fltLine = map(float, curLine)
		dataMat.append(fltLine)
	return dataMat
	
	
def distEclud(vecA, vecB):
	return sqrt(sum(power(vecA - vecB, 2)))
	
	
def randCent(dataSet, k):
	n = shape(dataSet)[1]
	centroids = mat(zeros((k, n)))
	for j in range(n):
		minJ = min(dataSet[:,j])
		rangeJ = float(max(dataSet[:,j]) - minJ)
		centroids[:,j] = minJ + rangeJ * random.rand(k, 1)
	return centroids
	
	
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
	m = shape(dataSet)[0]
	clusterAssment = mat(zeros((m, 2)))
	centroids = createCent(dataSet, k)
	clusterChanged = True
	while clusterChanged:
		clusterChanged = False
		for i in range(m):
			minDist = inf; minIndex = -1
			for j in range(k):
				# print 'i =',i,'j =', j
				distJI = distMeas(dataSet[i,:], centroids[j,:])
				if distJI<minDist:
					minDist = distJI
					minIndex = j
			if clusterAssment[i,0]!=minIndex:
				clusterChanged = True
				clusterAssment[i,:] = minIndex, minDist**2
		print centroids
		for cent in range(k):
			ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
			centroids[cent,:] = mean(ptsInClust, axis=0)
	return centroids, clusterAssment

	
def biKmeans(dataSet, k, distMeas=distEclud):
	m = shape(dataSet)[0]
	clusterAssment = mat(zeros((m, 2)))
	centroid0 = mean(dataSet, axis=0).tolist()[0]
	centList = [centroid0]
	for j in range(m):
		clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
	while (len(centList) < k):
		lowestSSE = inf
		for i in range(len(centList)):
			ptsInCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]
			centroidMat, splitClustAss = kMeans(ptsInCluster, 2, distMeas)
			sseSplit = sum(splitClustAss[:,1])
			sseNotSplit = sum(
				clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1]
			)
			print 'sseSplit =', sseSplit
			print 'sseNotSplit =', sseNotSplit
			if sseSplit+sseNotSplit < lowestSSE:
				lowestSSE = sseSplit + sseNotSplit
				bestCentToSplit = i
				bestNewCents = centroidMat
				bestClustAss = splitClustAss.copy()
		bestClustAss[nonzero(bestClustAss[:,0].A==1)[0], 0] = len(centList)
		bestClustAss[nonzero(bestClustAss[:,0].A==0)[0], 0] = bestCentToSplit
		print 'the bestCentToSplit is: ', bestCentToSplit
		print 'the len of bestClustAss is: ', len(bestClustAss)
		centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]
		centList.append(bestNewCents[1,:].tolist()[0])
		clusterAssment[nonzero(clusterAssment[:,0].A==bestCentToSplit)[0],:] = bestClustAss
	return mat(centList), clusterAssment

def showPlt(dataSet, centList, clusterAssment):
	m, n = shape(dataSet)
	patternList = ['ro', 'bs', 'g^']
	for i in range(m):
		pattern = patternList[int(clusterAssment[i,0])]
		plt.plot(dataSet[i,0], dataSet[i,1], pattern)
	plt.plot(centList[:,0], centList[:,1], 'yo')
	plt.show()
	
if __name__ == '__main__':
	# dataSet = loadDataSet('testSet.txt')
	# dataMat = mat(dataSet)
	# print 'dataMat =', dataMat
	# myCentroids, clusterAssing = kMeans(dataMat, 4)
	# print 'myCentroids =', myCentroids
	# print 'clusterAssing =', clusterAssing
	dataMat3 = mat(loadDataSet('testSet2.txt'))
	centList, clusterAssment = biKmeans(dataMat3, 3)
	print 'centList =', centList
	print 'clusterAssment =', clusterAssment
	showPlt(dataMat3, centList, clusterAssment)