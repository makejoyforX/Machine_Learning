#!/usr/bin/env python

import os
import sys
import time
import pickle
import logging
import random
import threading
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.cross_validation import KFold
from otto_loadData import load_otto
from otto_thread import otto_thread

def initLog(lda):
	# os.chdir(parentDirPath)
	slbda = str(lda)+'0000'
	slbda = slbda[:4].replace('.', '')
	logging.basicConfig(
				level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename=__file__[:-3]+'_'+slbda+'.log',
                filemode='w')


def testLogistic(otto, lbda=1.0, n_components=20, kbest=4):
	# X = otto.data[:1000, :20]
	# y = otto.target[:1000]
	X = otto.data[:, :]
	y = otto.target[:]
	# n_components = 20
	# kbest = 4
#	print 'y.shape =', y.shape

	scalar = StandardScaler().fit(X)
	X = scalar.transform(X)

	pca = PCA(n_components=n_components)
	selection = SelectKBest(k=kbest)

	combined_features = FeatureUnion(
		[("pca", pca), ('univ_select', selection)]
	)
	X_features = combined_features.fit(X,y).transform(X)

	logistic = LogisticRegression(C=1.0/lbda)
	pipes = [
		Pipeline(steps=[('features', combined_features), ('logistic', logistic)])\
		for i in range(5)
	]
	cv = KFold(n=X.shape[0], n_folds=5, shuffle=True)

	threadList = []
	for i,(trainIndex, testIndex) in enumerate(cv):
		pipe = pipes[i]
		trainData = X[trainIndex]
		trainTarget = y[trainIndex]
		# print trainTarget
		testData = X[testIndex]
		testTarget = y[testIndex]
		# pipe.fit(trainData, trainTarget)
		t = otto_thread(
			name=str(i+1), args=(pipe,), kwargs={'train':(trainData, trainTarget), 'test':(testData, testTarget)}
		)
		t.start()
		threadList.append(t)

	for t in threadList:
		t.join()

	# print pipe.fit(X, y)

if __name__ == '__main__':
	print 'begin...'
	print os.curdir

	# print otto.target[:10]
	lbdaList = [0.03, 0.1, 0.3, 1.0, 1.5, 2.0]
	# for lbda in lbdaList[-1]:
	if len(sys.argv)>1:
		lbda = float(sys.argv[1])
	else:
		lbda = 0.1
	if len(sys.argv)>2:
		n_components = int(sys.argv[2])
	else:
		n_components = 40
	initLog(lbda)
	otto = load_otto()
	logging.debug('working on logisitc, lambda=%f' % lbda)
	start_clock = time.clock()
	testLogistic(otto, lbda, n_components=40)
	end_clock = time.clock()
	logging.debug(str(end_clock - start_clock))
	

