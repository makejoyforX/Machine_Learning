from numpy import *
from numpy import linalg as la

def loadExData():
    return[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]
    
def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

def test1():
	U, Sigma, VT = linalg.svd([[1, 1], [7, 7]])
	print 'U =', U
	print 'Sigma =', Sigma
	print 'VT =', VT

	
def eulidSim(inA, inB):
	return 1.0 / (1.0 + la.norm(inA-inB))
	
def pearSim(inA, inB):
	if len(inA)<3:
		return 1.0
	return 0.5 + 0.5 * corrcoef(inA, inB, rowvar=0)[0][1]
	
def cosSim(inA, inB):
	num = float(inA.T * inB)
	denom = la.norm(inA) * la.norm(inB)
	return 0.5 + 0.5 * (num/denom)
	
def standEst(dataMat, user, simMeas, item):
	n = shape(dataMat)[1]
	simTotal = 0.0
	ratSimTotal = 0.0
	for j in range(n):
		userRating = dataMat[user, j]
		if userRating == 0:
			continue
		overLap = nonzero(logical_and(dataMat[:, item].A>0, dataMat[:, j].A>0))[0]
		# print 'overLap =', overLap
		# print 'logical_and =', logical_and(dataMat[:, item].A>0, dataMat[:, j].A>0)
		if len(overLap) == 0:
			similarity = 0.
		else:
			similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])
		# print 'the %d and %d similarity is: %f' % (item, j, similarity)
		simTotal += similarity
		ratSimTotal += similarity * userRating
	if simTotal == 0.:
		return 0.
	else:
		return ratSimTotal / simTotal

		
def svdEst(dataMat, user, simMeas, item):
	n = shape(dataMat)[1]
	simTotal = 0.0
	ratSimTotal = 0.0
	U, Sigma, VT = la.svd(dataMat)
	Sig4 = mat(eye(4) * Sigma[:4])
	xformedItems = dataMat.T * U[:,:4] * Sig4.I
	for j in range(n):
		userRating = dataMat[user, j]
		if userRating == 0 or j==item:
			continue
		similarity = simMeas(xformedItems[item, :].T, dataMat[j, :].T)
		# print 'the %d and %d similarity is: %f' % (item, j, similarity)
		simTotal += similarity
		ratSimTotal += similarity * userRating
	if simTotal == 0.:
		return 0.
	else:
		return ratSimTotal / simTotal
	


def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
	unratedItems = nonzero(dataMat[user,:].A == 0)[1]
	if len(unratedItems) == 0:
		return 'you rated everything'
	itemScores = []
	for item in unratedItems:
		# print 'item =', item
		estimatedScore = estMethod(dataMat, user, simMeas, item)
		itemScores.append((item, estimatedScore))
	return sorted(itemScores, key=lambda jj:jj[1], reverse=True)[:N]
	
	
if __name__ == '__main__':
	# test1()
	# set_printoptions(precision=4, suppress=True)
	# Data = loadExData()
	# U, Sigma, VT = linalg.svd(Data)
	# print 'Sigma =', Sigma
	# Sig3 = mat([[Sigma[0], 0, 0], [0, Sigma[1], 0], [0, 0, Sigma[2]]])
	# print U[:,:3] * Sig3 * VT[:3,:]
	# myMat = mat(loadExData())
	# print 'eulidSim(myMat[:,0], myMat[:,4]) =', eulidSim(myMat[:,0], myMat[:,4])
	# print 'eulidSim(myMat[:,0], myMat[:,0]) =', eulidSim(myMat[:,0], myMat[:,0])
	# print 'pearSim(myMat[:,0], myMat[:,4]) =', pearSim(myMat[:,0], myMat[:,4])
	# print 'pearSim(myMat[:,0], myMat[:,0]) =', pearSim(myMat[:,0], myMat[:,0])
	# print 'cosSim(myMat[:,0], myMat[:,4]) =', cosSim(myMat[:,0], myMat[:,4])
	# print 'cosSim(myMat[:,0], myMat[:,0]) =', cosSim(myMat[:,0], myMat[:,0])
	# myMat = mat(loadExData())
	# myMat[0, 1] = myMat[0, 0] = myMat[1, 0] = myMat[2, 0] = 4
	# myMat[3, 3] = 2
	# print myMat
	# print recommend(myMat, 2)
	myMat = mat(loadExData2())
	print recommend(myMat, 1, estMethod=svdEst)
	
	
	