import numpy as np
import matplotlib.pyplot as plt

import logging
import sys

class tree(object):
	
	def __init__(self, max_depth):
		'binary spit'
		self.max_depth = max_depth
		self.maxn = 2**max_depth
		self.nodes = [node(-1, -1) for i in xrange(self.maxn)]
		
	def __getitem__(self, i):
		if i>0 and i<self.maxn:
			return self.nodes[i]
		else:
			raise KeyError, "tree with index %d out of maxn" % i
			return None
	
	
	def __setitem__(self, key, val):
		self.nodes[key].update(val)
	
	
		
class node(object):
	""" A simple stump (max_depth = 1 Decision Tree)
	
	Parameters:
	-----------
	fid:	index of chosen features
	threshVal:	threshold value
		
	"""
	def __init__(self, fid, threshVal):
		self.fid = fid
		self.threshVal = self.threshVal
		
		
	def update(self, fid, threshVal):
		self.fid = fid
		self.thresh = threshVal
	
	def __str__(self):
		return "fid = %d, threshVal = %d" % (self.fid, self.threshVal)
	
		
if __name__ == "__main__":
	stump stp(1, 2)