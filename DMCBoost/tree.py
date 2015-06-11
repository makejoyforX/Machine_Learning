import numpy as np
import matplotlib.pyplot as plt

import logging
import sys

class tree(object):
	
	def __init__(self, max_depth):
		'binary spit'
		self.max_depth = max_depth
		self.maxn = 2**max_depth
		self.nodes = [node(-1, -1, -1, -1) for i in xrange(self.maxn)]
		
		
	def __getitem__(self, i):
		if i>0 and i<self.maxn:
			return self.nodes[i].elements()
		else:
			raise KeyError, "tree with index %d out of maxn" % i
			return None
	
	
	def predict_aVec(self, xi, rt=1, depth=1):
		if depth==self.max_depth:
			fid, val, l_label, r_label = self.nodes[rt].elements()
			if xi[fid]==val:
				return l_label
			else:
				return r_label
		else:
			fid, val, l_label, r_label = self.nodes[rt].elements()
			if xi[fid]==val:
				return self.predict_aVec(xi, rt<<1, depth+1)
			else:
				return self.predict_aVec(xi, rt<<1|1, depth+1)
	
	
	def predict(self, X):
		if len(X.shape)==1:
			return self.predict_aVec(X)
		else:
			return np.array(map(lambda xi: self.predict_aVec(xi), X))
	
	
	def print_nodes(self):
		for nd in self.get_nodes():
			print nd
	
	
	def get_nodes(self):
		return self.nodes[1:]
	
	
	def __setitem__(self, key, val):
		self.nodes[key].update(*val)
	
	
		
class node(object):
	""" A simple stump (max_depth = 1 Decision Tree)
	
	Parameters:
	-----------
	fid:	index of chosen features
	threshVal:	threshold value
		
	"""
	def __init__(self, fid, threshVal, l_label=-1, r_label=-1):
		self.fid = fid
		self.threshVal = threshVal
		self.l_label = l_label
		self.r_label = r_label
		
	def update(self, fid, threshVal, l_label, r_label):
		self.fid = fid
		self.threshVal = threshVal
		self.l_label = l_label
		self.r_label = r_label
		
		
	def elements(self):
		return self.fid, self.threshVal, self.l_label, self.r_label
		
		
	def __str__(self):
		return "fid = %d, threshVal = %d, l = %d, r = %d" %\
				(self.fid, self.threshVal, self.l_label, self.r_label)
	
		
if __name__ == "__main__":
	tr = tree(2)
	tr[1] = (1, 2, 3, 4)
	tr[2] = (3, 4, 5, 6)
	tr[3] = (2, 6, 7, 8)
	# for node in tr.get_nodes():
		# print node
	x = np.arange(16).reshape(4, 4)
	y = tr.predict(x)
	print x
	print y
	