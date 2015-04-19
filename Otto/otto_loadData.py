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
		with open(os.path.join(fpath, desFileName), 'w') as fout:
			for line in lines:
				line = line.strip()
				intervals = line.split(',')[1:-1]
				label = intervals[-1].split('_')[-1]
				intervals.append(label)
				# intervals = map(float, intervals)
				# ss = ','.join(map(str, intervals))
				# print ss
				fout.write(','.join(intervals)+'\n')


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
	# load_otto()
	load_rawData(fname=rawTestFileName)
	load_otto()
