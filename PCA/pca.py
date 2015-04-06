from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName, delim='\t'):
	fr = open(fileName)
	stringArr = [line.strip().split(delim) for line in fr.readlines()]
	dataArr = [map(float, line) for line in stringArr]
	return mat(dataArr)
	
def pca(dataMat, topNfeat=9999999):
	meanVals = mean(dataMat, axis=0)
	meanRemoved = dataMat - meanVals
	covMat = cov(meanRemoved, rowvar=0)
	eigVals, eigVects = linalg.eig(mat(covMat))
	eigValInd = argsort(eigVals)
	eigValInd = eigValInd[:-(topNfeat+1):-1]
	redEigVects = eigVects[:, eigValInd]
	lowDDataMat = meanRemoved * redEigVects
	reconMat = (lowDDataMat * redEigVects.T) + meanVals
	return lowDDataMat, reconMat
	

def showPlot(dataMat, reconMat):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	# print 'shape(dataMar) =', shape(dataMat)
	# print 'dataMat =', dataMat
	# print 'reconMat =', reconMat
	ax.scatter(dataMat[:,0].flatten().A[0], dataMat[:,1].flatten().A[0], marker='^', s=90, c='blue')
	ax.scatter(reconMat[:,0].flatten().A[0], reconMat[:,1].flatten().A[0], marker='o', s=50, c='red')
	plt.show()
	
	
def replaceNonWithMean(fpath):
	dataMat = loadDataSet(fpath, ' ')
	numFeat = shape(dataMat)[1]
	for i in range(numFeat):
		meanVal = mean(dataMat[nonzero(~isnan(dataMat[:,i].A))[0], i])
		dataMat[nonzero(isnan(dataMat[:,i].A))[0], i] = meanVal
	return dataMat
	
if __name__ == '__main__':
	# dataMat = loadDataSet('f:/ML/Machine_Learning/PCA/testSet.txt')
	# lowDMat, reconMat = pca(dataMat, 2)
	# print 'shape(lowDMat) =', shape(lowDMat)
	# print 'shape(reconMat) =', shape(reconMat)
	# showPlot(dataMat, reconMat)
	fpath = 'f:/ML/Machine_Learning/PCA/secom.data'
	dataMat = replaceNonWithMean(fpath)
	lowDMat, reconMat = pca(dataMat, 1)
	showPlot(dataMat, reconMat)
	