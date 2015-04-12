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
			vectors.append(list(struct.unpack('B'*n_size, fin.read(struct.calcsize('B'*n_size)))))
		# print vectors[:10]
	flat_data = np.array(vectors)
		
	with open(join(fpath, fLabelName), 'rb') as fin:
		magicNumber, n_sample = struct.unpack('>II', fin.read(struct.calcsize("=II")))
		# print 'magicNumber =', magicNumber
		# print 'n_sample =', n_sample
		# pres = struct.unpack('bbbbbbbb', fin.read(struct.calcsize("=bbbbbbbb")))
		# print 'pres =', pres
		labels = struct.unpack('B'*n_sample, fin.read(struct.calcsize('B'*n_sample)))
		# print labels[:10]
		# print len(labels), labels[0]
	target = np.array(labels)
	images = flat_data.view()
	images.shape = (-1, n_row, n_col)
	
	# for index, (image, prediction) in enumerate(list(zip(flat_data[:4], target[:4]))):
		# plt.subplot(1, 4, index+1)
		# plt.axis('off')
		# imageData = image.reshape(28,28)
		# plt.imshow(imageData, cmap=plt.cm.gray_r, interpolation='nearest')
		# plt.title('Prediction: %i' % prediction)
	# plt.show()
	
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
	
	
def save_myDigits(digits, fpath='F:/ML/digits/', fname='train.csv'):
	flat_data = digits.data
	label = digits.target
	# print label.shape
	# print flat_data.shape
	label = label.reshape(label.shape[0], 1)
	data = np.hstack((flat_data, label))
	np.savetxt(join(fpath,fname), data, delimiter=',')
	
	
if __name__ == '__main__':
	digits = load_myDigits()
	# show_OneDigit(digits)
	values = (0x0801, 0x00ea60)
	raw_data = struct.pack('II', *values)
	print binascii.hexlify(raw_data)
	