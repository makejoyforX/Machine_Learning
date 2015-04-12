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
import logging
from sklearn import svm
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from access_digits import load_myDigits

def initLog():
	logging.basicConfig(
				level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='digits_svm.log',
                filemode='w')

				
def testSVM(trainDigits, testDigits):
	clf = svm.SVC(C=1.0, cache_size=200, gamma=0.001)
	clf.fit(trainDigits.data, trainDigits.target)
	logging.debug('finish logistic fit')
	logging.debug('svm accuracy is %.3f' % clf.score(testDigits.data, testDigits.target))
	# predicted = clf.predict(testDigits.data)
	# logging.debug('svm accuracy is %.3f' % clf.score(trainDigits.data[:100], trainDigits.target[:100]))
	# predicted = clf.predict(trainDigits.data[:100])
	
	# for index, (image, prediction) in enumerate(list(zip(trainDigits.data[:4], trainDigits.target[:4]))):
		# plt.subplot(2, 4, index+1)
		# plt.axis('off')
		# imageData = image.reshape(28,28)
		# plt.imshow(imageData, cmap=plt.cm.gray_r, interpolation='nearest')
		# plt.title('Prediction: %i' % prediction)
		
	# for index, (image, prediction) in enumerate(list(zip(testDigits.data[:4], predicted[:4]))):
	# for index, (image, prediction) in enumerate(list(zip(trainDigits.data[:4], predicted[:4]))):
		# plt.subplot(2, 4, index+5)
		# plt.axis('off')
		# imageData = image.reshape(28,28)
		# plt.imshow(imageData, cmap=plt.cm.gray_r, interpolation='nearest')
		# plt.title('Prediction: %i' % prediction)
		
	# print predicted
	# print testDigits.target[:100]
	
def usingPCA(trainDigits, testDigits):
	clf = svm.SVC(C=1.0, cache_size=200, gamma=0.001)
	pca = PCA(n_components=15*15).fit(trainDigits.data)
	train_Trans = pca.transform(trainDigits.data)
	clf.fit(train_Trans, trainDigits.target)
	test_Trans = pca.transform(testDigits.data)
	logging.debug('finish logistic fit')
	logging.debug('svm(after PCA) accuracy is %.3f' % clf.score(test_Trans, testDigits.target[:100]))
	
	
if __name__ == '__main__':
	initLog()
	logging.debug("Working on svm")
	
	trainDigits = load_myDigits()
	testDigits = load_myDigits(fDataName='t10k-images-idx3-ubyte', fLabelName='t10k-labels-idx1-ubyte')
	
	start = time.clock()
	testSVM(trainDigits, testDigits)
	end = time.clock()
	logging.debug('testSVM without PCA:'+str(end - start))
	
	start = time.clock()
	usingPCA(trainDigits, testDigits)
	end = time.clock()
	logging.debug('testSVM with PCA:'+str(end - start))
	
	plt.show()
	
	