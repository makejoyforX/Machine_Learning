#!/usr/bin/env python

import sys
import logging
import os
import threading
import datetime
import time
import gc

import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.metrics import zero_one_loss, accuracy_score
from sklearn.cross_validation import cross_val_predict
from sklearn.cross_validation import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

class clf_thread(threading.Thread):

	def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
		threading.Thread.__init__(self, group=group, target=target, name=name, verbose=verbose)
		self.args = args
		self.kwargs = kwargs
		return 

		
	def run(self):
		if len(self.args)>0 and 'test' in self.kwargs and 'train' in self.kwargs:
			clf = self.args[0]
			logging.debug('threadName = %s, begin...' % (self.name))
			clf.fit(*self.kwargs['train'])
			X_test, y_test = self.kwargs['test']
			y_pred = clf.predict(X_test)
			error_rate = 1 - accuracy_score(y_test, y_pred)
			logging.debug('%s: [%s] error rate is %.6f' % (self.name, self.kwargs['algo'], error_rate))
			logging.debug('threadName = %s, end...' % (self.name))
		return

		

def init_Logging():
	tt = datetime.date.today()
	fname = "%s_V%02d%02d.log" % (__file__[:-3], tt.month, tt.day)
	logging.basicConfig(
				level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename=fname,
                filemode='w')


def predict_adaBoost(X, y, algorithm):
	max_depth = 3
	n_estimators = 500
	dt_clf = DecisionTreeClassifier(max_depth=max_depth)
	algorithm = algorithm
	n_folds = 5
	learning_rate = 0.1

	cv = KFold(n=X.shape[0], n_folds=n_folds, shuffle=True)
	
	threadList = []
	
	for i, (trainIdx, testIdx) in enumerate(cv):
		clf = AdaBoostClassifier(
			base_estimator=dt_clf,
			n_estimators=n_estimators,
			learning_rate=learning_rate,
			algorithm=algorithm,
		)
		trainData = X[trainIdx]
		trainTarget = y[trainIdx]
		testData = X[testIdx]
		testTarget = y[testIdx]
		t = clf_thread(
			name=str(i+1), args=(clf, ),
			kwargs={'algo':algorithm, 'train':(trainData, trainTarget), 'test':(testData, testTarget)}
		)
		t.start()
		threadList.append(t)
		
	for t in threadList:
		t.join()
		
		
def predict_samme(X, y):
	predict_adaBoost(X, y, "SAMME")
	
	
def predict_samme_r(X, y):
	predict_adaBoost(X, y, "SAMME.R")

	
def predict_GDMCBoost(X, y):
	max_depth = 3
	n_estimators = 500
	n_folds = 5
	learning_rate=0.1

	cv = KFold(n=X.shape[0], n_folds=n_folds, shuffle=True)
	
	threadList = []
	
	for i, (trainIdx, testIdx) in enumerate(cv):
		clf = GradientBoostingClassifier(
			n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate
		)
		trainData = X[trainIdx]
		trainTarget = y[trainIdx]
		testData = X[testIdx]
		testTarget = y[testIdx]
		t = clf_thread(
			name=str(i+1), args=(clf, ),
			kwargs={'algo':"GDMCBoost", 'train':(trainData, trainTarget), 'test':(testData, testTarget)}
		)
		t.start()
		threadList.append(t)
		# clf.fit(trainData, trainTarget)
		# predTarget = clf.predict(testData)
		# error_rate = 1 - accuracy_score(testTarget, predTarget)
		# logging.debug('%s: [%s] error rate is %.6f' % (str(i+1), "GDMCBoost", error_rate))

	for t in threadList:
		t.join()


def predict_xgBoost(X, y):
	max_depth = 3
	n_estimators = 500
	learning_rate = 0.1
	silent = True
	nthread = -1
	n_folds = 5

	cv = KFold(n=X.shape[0], n_folds=n_folds, shuffle=True)
	
	for i, (trainIdx, testIdx) in enumerate(cv):
		clf = xgb.XGBClassifier(
			max_depth=max_depth, n_estimators=n_estimators,
			learning_rate=learning_rate, silent=silent, nthread=nthread
		)
		trainData = X[trainIdx]
		trainTarget = y[trainIdx]
		testData = X[testIdx]
		testTarget = y[testIdx]
		clf.fit(trainData, trainTarget)
		predTarget = clf.predict(testData)
		error_rate = 1 - accuracy_score(testTarget, predTarget)
		logging.debug('%s: [%s] error rate is %.6f' % (str(i+1), "xgBoost", error_rate))
		
		
def load_data(fname, dtype):
	data = np.loadtxt(fname, dtype=dtype, delimiter=',')
	X = data[:, :-1]
	y = data[:, -1]
	return X, y
	
	
def begin_test():
	init_Logging()
	d_fpath = "./ddata"
	ddata = [
		"wine.data",
		"car.data",
		"CNAE-9.data",
		"nursery.data",
		"agaricus.data",
		"letter.data",
		"poker.data",
	]
	for fname in ddata:
		X, y = load_data(os.path.join(d_fpath, fname), np.int32)
		logging.debug("[%s] (%d x %d) begin AdaBoost ..." % (fname, X.shape[0], X.shape[1]))
		start_clock = time.clock()
		predict_samme(X, y)
		end_clock = time.clock()
		logging.debug("[%s] (%.2fs) end AdaBoost ..." % (fname, end_clock-start_clock))
		
		logging.debug("[%s] (%d x %d) begin GDMCBoost ..." % (fname, X.shape[0], X.shape[1]))
		start_clock = time.clock()
		if fname != ddata[-1]:
			predict_GDMCBoost(X, y)
		end_clock = time.clock()
		logging.debug("[%s] (%.2fs) end GDMCBoost ..." % (fname, end_clock-start_clock))
		
		logging.debug("[%s] (%d x %d) begin xgBoost ..." % (fname, X.shape[0], X.shape[1]))
		start_clock = time.clock()
		predict_xgBoost(X, y)
		end_clock = time.clock()
		logging.debug("[%s] (%.2fs) end xgBoost ..." % (fname, end_clock-start_clock))
		
		logging.debug("\n")
		# collect garbage
		gc.collect()

		
if __name__ == "__main__":
	begin_test()
	
