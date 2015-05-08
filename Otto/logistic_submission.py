#!/usr/bin/env python

import os
import sys
import time
import pickle
import logging
import random
import threading
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.cross_validation import KFold
from otto_loadData import load_otto, load_testotto
from otto_thread import otto_thread

def initLog(logname):
	# os.chdir(parentDirPath)
	logging.basicConfig(
				level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename=logname,
                filemode='w')


def testLogistic(lbda=1.0, n_components=20, kbest=4):
	# X = otto.data[:1000, :20]
	# y = otto.target[:1000]
	otto = load_otto()
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
	pipe = Pipeline(steps=[('features', combined_features), ('logistic', logistic)])
	trainData = X
	trainTarget = y
	pipe.fit(trainData, trainTarget)
	# print trainTarget
	test_otto = load_testotto()
	testData = test_otto.data
	testData = scalar.transform(testData)
	# logging.debug('lambda=%.3f: score is %.3f' % (lbda, pipe.score()))
	'save the prediction'
	prediction = pipe.predict_proba(testData)
	proba = pipe.predict_proba(testData)
	save_submission(lbda, proba, prediction)


def save_submission(lbda, proba, prediction):
	slbda = (str(lbda)+"0000")[:4].replace('.', '')
	sub_fname = 'zyx_otto_submission_%s.csv' % (slbda)
	# m = proba.shape[0]
	# ids = np.arange(1, m+1).reshape(-1,1)
	# entry = np.hstack((ids, proba))
	# header = 'id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9'
	entry = proba
	fmt = '%.5f'
	np.savetxt(sub_fname, entry, fmt=fmt, delimiter=',')
	res_fname = 'zyx_otto_result_%s.csv' % (slbda)
	m = prediction.shape[0]
	ids = np.arange(1, m+1).reshape(-1,1)
	entry = np.hstack((ids, proba))
	np.savetxt(res_fname, entry, delimiter=',')



if __name__ == '__main__':
	if len(sys.argv)>1:
		lbda = float(sys.argv[1])
	else:
		lbda = 1.00
	if len(sys.argv)>2:
		n_components = int(sys.argv[2])
	else:
		n_components = 40	
	logname = 'otto_sub.log'
	initLog(logname)
	testLogistic(lbda=lbda, n_components=n_components)
	