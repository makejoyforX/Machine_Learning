#!/usr/bin/env python

import os
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

def initLog():
	# os.chdir(parentDirPath)
	logging.basicConfig(
				level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename=__file__[:-2]+'log',
                filemode='w')


def testLogistic(otto, lbda=1.0):
	X = otto.data[:100, :10]
	y = otto.target[:100]
	# X = otto.data
	# y = otto.target
	n_components = 5
	kbest = 1
#	print 'y.shape =', y.shape

	scalar = StandardScaler().fit(X)
	X = scalar.transform(X)

	pca = PCA(n_components=n_components)
	selection = SelectKBest(k=kbest)

	combined_features = FeatureUnion(
		[("pca", pca), ('univ_select', selection)]
	)
	X_features = combined_features.fit(X,y).transform(X)

	logistic = LogisticRegression(C=1/ldba)
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
	initLog()
	otto = load_otto()
	# print otto.target[:10]
	lbdaList = [0.03, 0.1, 0.3, 1.0, 1.5]
	for lbda in lbdaList:
		logging.debug('working on logisitc, lambda=%f' % lbda)
		start_clock = time.clock()
		testLogistic(otto, lbda)
		end_clock = time.clock()
		logging.debug(str(end_clock - start_clock))
	

