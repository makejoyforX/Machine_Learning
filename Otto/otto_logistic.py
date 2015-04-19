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


def testLogistic(otto):
#	X = otto.data[:20000, :20]
#	y = otto.target[:20000]
	X = otto.data
	y = otto.target

#	print 'y.shape =', y.shape

	scalar = StandardScaler().fit(X)
	X = scalar.transform(X)

	pca = PCA(n_components=20)
	selection = SelectKBest(k=4)

	combined_features = FeatureUnion(
		[("pca", pca), ('univ_select', selection)]
	)
	X_features = combined_features.fit(X,y).transform(X)

	logistic = LogisticRegression()
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
	logging.debug('working on logisitc')
	start_clock = time.clock()
	testLogistic(otto)
	end_clock = time.clock()
	logging.debug(str(end_clock - start_clock))

