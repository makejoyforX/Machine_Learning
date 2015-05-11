#!/usr/bin/env python

import os
import sys
import time
import logging
import random
import threading
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.cross_validation import KFold
from otto_loadData import load_otto
from otto_thread import otto_thread

def initLog(lda):
	slbda = str(lda) + '0000'
	slbda = slbda[:4].replace('.', '')
	logging.basicConfig(
		level=logging.DEBUG,
		format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=__file__[:-3]+'_'+slbda+'.log',
        filemode='w'
	)

def testSVC(otto, lbda=1.0, n_components=20, kbest=4):
	X = otto.data
	y = otto.target
	# X = otto.data[:, :40]
	# y = otto.target[:]

	scalar = StandardScaler().fit(X)
	X = scalar.transform(X)

	pca = PCA(n_components=n_components)
	selection = SelectKBest(k=kbest)

	combined_features = FeatureUnion(
		[("pca", pca), ('univ_select', selection)]
	)
	X_features = combined_features.fit(X,y).transform(X)

	svc = SVC(C=1.0/lbda, kernel='rbf', cache_size=300)
	n_folds = 4
	pipes = [
		Pipeline(steps=[('features', combined_features), ('svc', svc)])\
		for i in range(n_folds)
	]
	cv = KFold(n=X.shape[0], n_folds=16, shuffle=True)

	threadList = []
	for i,(trainIndex, testIndex) in enumerate(cv):
		if i >= n_folds:
			break
		pipe = pipes[i]
		trainData = X[trainIndex]
		trainTarget = y[trainIndex]
		testData = X[testIndex]
		testTarget = y[testIndex]
		t = otto_thread(
			name=str(i+1), args=(pipe,), kwargs={'algo':'SVC', 'train':(trainData, trainTarget), 'test':(testData, testTarget)}
		)
		t.start()
		threadList.append(t)

	for t in threadList:
		t.join()


if __name__ == '__main__':
	print 'begin...'

	# print otto.target[:10]
	lbdaList = [0.03, 0.1, 0.3, 1.0, 1.5, 2.0]
	# for lbda in lbdaList[-1]:
	if len(sys.argv)>1:
		lbda = float(sys.argv[1])
	else:
		lbda = 1.0
	if len(sys.argv)>2:
		n_components = int(sys.argv[2])
	else:
		n_components = 30
	initLog(lbda)
	otto = load_otto()
	logging.debug('working on SVC, lambda=%f' % lbda)
	start_clock = time.clock()
	testSVC(otto, lbda, n_components=n_components, kbest=4)
	end_clock = time.clock()
	logging.debug(str(end_clock - start_clock))
