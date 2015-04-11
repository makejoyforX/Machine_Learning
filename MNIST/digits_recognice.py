import os
import csv
import shutil
from os import environ
from os.path import dirname
from os.path import join
from os.path import exists
from os.path import expanduser
from os.path import isdir
from os import listdir
from os import makedirs
import time
import struct
import binascii
from sklearn import linear_model
from sklearn.datasets import base
import numpy as np
import matplotlib.pyplot as plt

def load_myDigits(fpath='F:/ML/digits/', fDataName='train-images-idx3-ubyte', fLabelName='train-labels-idx1-ubyte'):
	with open(join(fpath, fDataName), 'rb') as fin:
		magicNumber, n_sample, n_row, n_col = struct.unpack('>IIII', fin.read(struct.calcsize("=IIII")))
		n_size = n_row * n_col
		vectors = []
		for i in xrange(n_sample):
			vectors.append(list(struct.unpack('b'*n_size, fin.read(struct.calcsize('b'*n_size)))))
	flat_data = np.array(vectors)
		
	with open(join(fpath, fLabelName), 'rb') as fin:
		magicNumber, n_sample = struct.unpack('>II', fin.read(struct.calcsize("=II")))
		print 'magicNumber =', magicNumber
		print 'n_sample =', n_sample
		# pres = struct.unpack('bbbbbbbb', fin.read(struct.calcsize("=bbbbbbbb")))
		# print 'pres =', pres
		labels = struct.unpack('b'*n_sample, fin.read(n_sample))
		# print len(labels), labels[0]
	target = np.array(labels)
	images = flat_data.view()
	images.shape = (-1, n_row, n_col)
	
	return base.Bunch(
		data=flat_data,
		target=target.astype(np.int),
		target_names=np.arange(10),
		images=images
	)

def show_OneDigit(digits, index=0):
	if index>=digits.images.shape[0]:
		print 'index of out of %d' % (images.shape[0])
		return
	import pylab as pl
	print 'label =', digits.target[index]
	pl.gray()
	pl.matshow(digits.images[index])
	pl.show()
	
def testLogistic():
	trainDigits = load_myDigits()
	testDigits = load_myDigits(fDataName='t10k-images-idx3-ubyte', fLabelName='t10k-labels-idx1-ubyte')
	logistic = linear_model.LogisticRegression()
	logistic.fit(trainDigits.data, trainDigits.target)
	print 'finish logistic fit'
	print 'logistic accuracy is %.3f' % logistic.score(testDigits.data, testDigits.target)
	
	
if __name__ == '__main__':
	# digits = load_myDigits()
	# show_OneDigit(digits)
	# values = (0x0801, 0x00ea60)
	# raw_data = struct.pack('II', *values)
	# print binascii.hexlify(raw_data)
	start_clock = time.clock()
	testLogistic()
	end_clock = time.clock()
	print end_clock - start_clock
	# time = 5585s