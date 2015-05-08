#!/usr/bin/env python

import os.path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import base


parentDirPath = '/home/turf/Code/Machine_Learning/Otto'
rawTrainFileName = 'rawtrain.csv'
rawTestFileName  = 'rawtest.csv'
trainFileName = 'train.csv'
testFileName  = 'test.csv'


def load_rawData(fname=rawTrainFileName, fpath=parentDirPath):
	with open(os.path.join(fpath, fname), 'r') as fin:
		lines = fin.readlines()[1:]
		desFileName = fname[3:]
		# labelSet = set()
		# for line in lines[-1:]:
		# 		line = line.strip()
		# 		intervals = line.split(',')[1:]
		# 		label = intervals[-1].split('_')[-1]
		# 		labelSet.add(int(label))
		# 		intervals[-1] = label
		with open(os.path.join(fpath, desFileName), 'w') as fout:
			for line in lines:
				line = line.strip()
				intervals = line.split(',')[1:]
				label = intervals[-1].split('_')[-1]
				labelSet.add(int(label))
				intervals[-1] = label
				intervals = map(float, intervals)
				ssq = ','.join(map(str, intervals))+'\n'
				fout.write(ssq)
		# print 'labelSet =\n', labelSet


def load_testotto(fname=testFileName, fpath=parentDirPath):
	data = np.loadtxt(os.path.join(fpath, fname), delimiter=',', dtype=float)
	flat_data = data[:,:]
	images = flat_data.view()
	return base.Bunch(
		data=flat_data,
		target=None,
		target_names=np.arange(1,10),
		images=images,
	)

def load_otto(fname=trainFileName, fpath=parentDirPath):

	'''load_data from fname'''
	data = np.loadtxt(os.path.join(fpath, fname), delimiter=',', dtype=float)
	flat_data = data[:,:-1]
	target = data[:,-1]
	images = flat_data.view()
	return base.Bunch(
		data=flat_data,
		target=target.astype(np.int),
		target_names=np.arange(1,10),
		images=images,
	)

	# with open(os.path.join(fpath, 'otto.log'), 'w') as fout:
	# 	fout.write(str(data)+'\n')



if __name__ == '__main__':
	# load_rawData()
	load_rawData(fname=rawTestFileName)
	# otto = load_otto()
	# otto = load_testotto()
	# print otto.data.shape