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
from sklearn import linear_model
from sklearn.datasets import base
import numpy as np
import matplotlib.pyplot as plt
from access_digits import load_myDigits

def initLog():
	logging.basicConfig(
				level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='digits_logistic.log',
                filemode='w')
				
				
def testLogistic():
	trainDigits = load_myDigits()
	testDigits = load_myDigits(fDataName='t10k-images-idx3-ubyte', fLabelName='t10k-labels-idx1-ubyte')
	logistic = linear_model.LogisticRegression()
	logistic.fit(trainDigits.data[:10000], trainDigits.target[:10000])
	logging.debug('finish logistic fit')
	logging.debug('logistic accuracy is %.3f' % logistic.score(testDigits.data[:500], testDigits.target[:500]))
	# logistic.fit(trainDigits.data[:1000], trainDigits.target[:1000])
	# logging.debug('finish logistic fit')
	# logging.debug('logistic accuracy is %.3f' % logistic.score(testDigits.data[:100], testDigits.target[:100]))
	
	# predicted = logistic.predict(testDigits.data[:100])
	# for index, (image, prediction) in enumerate(list(zip(testDigits.data[:4], predicted[:4]))):
		# plt.subplot(2, 4, index+1)
		# plt.axis('off')
		# imageData = image.reshape(28,28)
		# plt.imshow(imageData, cmap=plt.cm.gray_r, interpolation='nearest')
		# plt.title('Prediction: %i' % prediction)
	# plt.show()	
	
if __name__ == '__main__':
	# digits = load_myDigits()
	# show_OneDigit(digits)
	# values = (0x0801, 0x00ea60)
	# raw_data = struct.pack('II', *values)
	# print binascii.hexlify(raw_data)
	initLog()
	logging.debug("Working on logistic")
	start_clock = time.clock()
	testLogistic()
	end_clock = time.clock()
	logging.debug(str(end_clock - start_clock))
	# time = 5585s