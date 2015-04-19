#!/usr/bin/env python

import numpy as np
import os

parentDirPath = '/home/turf/Code/Machine_Learning/Otto'

def saveMyCsv(fname='test.csv'):
	os.chdir(parentDirPath)
	a = [1., 2., 3., 4.]
	with open(fname, 'w') as fout:
		strOfa = map(str, a)
		print strOfa
		fout.write(','.join(strOfa)+'\n')

if __name__ == '__main__':
	# saveMyCsv()
	x = np.loadtxt(parentDirPath+'/test.csv', delimiter=',')
	print x
