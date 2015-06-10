#!/usr/bin/env python

import logging
import sys
import copy
from collections import defaultdict
from operator import itemgetter

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification, load_iris
from sklearn.metrics import accuracy_score, zero_one_loss
from tree import tree


class DMCBoost(object):

	def __init__(self, max_initial_round, max_margin_round, bn_rate, max_depth, eps=0.01, tol=1e-5):
		self.max_initial_round = max_initial_round
		self.max_round = max_initial_round + max_margin_round
		self.max_depth = max_depth
		self.bn_rate = bn_rate
		self.tol = tol
		self.eps = eps
		
		
	def fit(self, X, y):
		self.X = X
		self.y = y
		m, n = X.shape
		self.target_set = np.unique(y)
		self.feature_list = map(
			lambda i: list(np.unique(X[:, i])), xrange(n)
		)
		print self.feature_list
		self.initial_mode()
		#self.margin_maximization()
	
	
	def f_x_y(self):
		pass
		
	
	def	predict(self, X):
		#m = X.shape[0]
		ret_y = []
		cn = len(self.target_set)
		target_dict = dict(zip(self.target_set, range(cn)))
		rtarget_dict = dict(zip(range(cn), self.target_set))
		for x in X:
			yi_list = np.zeros(cn, dtype=np.float)
			for i,tr in enumerate(self.tree_list):
				alpha = self.alpha_list[i]
				pred_y = tr.predict_aVec(x)
				pred_y = 0 if pred_y < 0 else pred_y
				yi_list[target_dict[pred_y]] += alpha
			ret_y.append(rtarget_dict[yi_list.argmax()])
				
		return np.array(ret_y)
	
	
	def a_func_idx(self, idx, y):
		ret = 0.0
		for i,ht in enumerate(self.ht_list):
			if ht[idx]==y:
				ret += self.alpha_list[i] 
		return ret
	
	
	def get_l(self, ix, yi):
		"""
		 l = argmax_{y \in Y, y \neq yi} a(xi, y)
		"""
		mx_a = -sys.maxint
		mx_y = -1
		for y in self.target_set:
			if y != yi:
				a_xi_y = self.a_func_idx(ix, y)
				if a_xi_y > mx_a:
					mx_a = a_xi_y
					mx_y = y
		return mx_y
				

	def line_search_min_loss(self, ht, X, y):
		"""
		Algorithm 1.1
		
		parameters:
		------------
		ht:		   h_t() function
		y:		  y of the data
		l:		  current label
		
		return:
		minError:	 min Error
		interval:	 optimal interval has the min Error
		
		"""
		e_dict = defaultdict(int)
		m = X.shape[0]
		for i in xrange(m):
			if ht[i] == y[i]:
				l = self.get_l(i, y[i])
				q = self.a_func_idx(i, l) - self.a_func_idx(i, y[i])
				error_update = -1
			else:
				a_xi_yi = self.a_func_idx(i, y[i])
				a_xi_y = self.a_func_idx(i, ht[i])
				if a_xi_yi > a_xi_y:
					q = a_xi_y - a_xi_yi
					error_update = 1
				else:
					continue
			e_dict[q] += error_update
		if len(e_dict) == 0:
			return (float(-sys.maxint), float(sys.maxint)), 0.0
			
		intervals = sorted(e_dict.iteritems(), key=itemgetter(0))
		# incrementally calculate classification error
		mn_idx = -1
		mn_error = sys.maxint
		acc = 0
		length = len(intervals)
		# intervals.append((-1, 0))
		for i in xrange(length):
			acc += intervals[i][1]
			if acc < mn_error:
				mn_error = acc
				mn_idx = i
				
		if mn_idx == 0:
			ret_interval = (float(-sys.maxint), intervals[0][0])
		elif mn_idx == length-1:
			ret_interval = (intervals[mn_idx][0], float(sys.maxint))
		else:
			ret_interval = (intervals[mn_idx][0], intervals[mn_idx+1][0])
			
		return ret_interval, mn_error

	
	def get_argmin_loss(self, ht, idx, indice, X, y):
		mn_loss = sys.maxint
		mn_l = -1
		for l in self.target_set:
			'setting the input ht(x)=l if x \in S'
			ht[idx[indice]] = l
			# print 'before run min-loss ht =', ht
			_, loss = self.line_search_min_loss(ht, self.X, self.y)
			# print 'After run min-loss ht =', ht
			if loss < mn_loss:
				mn_l = l
				mn_loss = loss
		return mn_l, mn_loss
			
		
	def build_tree(self, tr, ht, idx, feature_list, depth, rt, get_argmin_loss):
		depth += 1
		if depth > self.max_depth:
			return 
		
		X = self.X[idx]
		y = self.y[idx]
		mn_loss = sys.maxint
		bst_feat, bst_feat_val = -1, -1
		bst_l_left, bst_l_right = -1, -1
		for feat, feat_val_set in enumerate(feature_list):
			for val in feat_val_set:
				'left equal, right not equal'
				indices = (X[:, feat] == val)
				X_left, y_left = X[indices], y[indices]
				X_right, y_right = X[~indices], y[~indices]
				if len(X_left)==0 or len(X_right)==0:
					continue
				l_left , loss_left	= get_argmin_loss(ht, idx, indices, X_left, y_left)
				ht[idx[indices]] = l_left
				l_right, loss_right = get_argmin_loss(ht, idx, ~indices, X_right, y_right)
				ht[idx[~indices]] = l_right
				if loss_left+loss_right < mn_loss:
					mn_loss = loss_left + loss_right
					bst_feat, bst_feat_val = feat, val
					bst_l_left, bst_l_right = l_left, l_right
		
		# add (bst_feat, bst_feat_val) to tree
		tr[rt] = (bst_feat, bst_feat_val, bst_l_left, bst_l_right)
		
		print 'bst_feat =', bst_feat, 'bst_feat_val =', bst_feat_val
		# print 'before remove =', feature_list[bst_feat]
		
		# delete bst_feat_val from bst_feat
		feature_list[bst_feat].remove(bst_feat_val)
		bst_indices = (X[:, bst_feat] == bst_feat_val)
		idx_left, idx_right = idx[bst_indices], idx[~bst_indices]
		ht[idx_left] = bst_l_left
		ht[idx_right] = bst_l_right
		
		# print 'After remove =', feature_list[bst_feat]
		
		self.build_tree(tr, ht, idx_left, feature_list, depth, rt<<1, get_argmin_loss)
		self.build_tree(tr, ht, idx_right, feature_list, depth, rt<<1|1, get_argmin_loss)
		
	
	def initial_mode(self):
		"""
		Phase I. Minimizing Multi-class Classification Errors.
		
		"""
		self.t_ = 0
		self.ht_list = []
		self.bst_error_list = []
		self.alpha_list = []
		self.tree_list = []
		m, n = self.X.shape
		idx = np.arange(m)
		while self.t_ < self.max_initial_round:
			ht = -np.ones(m, dtype=np.int)
			feature_list = copy.deepcopy(self.feature_list)
			ctree = tree(self.max_depth)
			self.build_tree(ctree, ht, idx, feature_list, 0, 1, self.get_argmin_loss)
			bst_interval, bst_error = self.line_search_min_loss(ht, self.X, self.y)
			alpha_t = (bst_interval[0] + bst_interval[1]) / 2.0
			self.ht_list.append(ht)
			self.bst_error_list.append(bst_error)
			self.alpha_list.append(alpha_t)
			self.tree_list.append(ctree)
			self.t_ += 1
			
			print "round %d:" % (self.t_)
			print 'bst_interval = %s, bst_error = %s' % (str(bst_interval), str(bst_error))

			
	def ft_x_y(self, idx, y):
		return self.a_func_idx(idx, y)
	
	
	def get_argmax_ft(self, ix, yi):
		mx_ft_x_y = float(-sys.maxint)
		mx_l = -1
		for y in self.y:
			if y == yi:
				continue
			ft_x_y = self.ft_x_y(self, ix)
			if ft_x_y > mx_ft_x_y:
				mx_ft_x_y = ft_x_y
				mx_l = y
		return mx_l, mx_ft_x_y
	
	
	def get_worst_M(self, idx):
		"""
		though we need to get the worst x_i, but we use idx instead.
		It may become much faster.
		"""
		Mi = []
		for ix in idx:
			ft_xi_yi = self.ft_x_y(ix, self.y[ix])
			_, mx_ft_xi_y = self.get_argmax_ft(ix, self.y[ix])
			Mi.append(ft_xi_yi - mx_ft_xi_y)
		return Mi
		
		
	def get_gavg_n_(self, idx, bn):
		mi = self.get_worst_M(idx)
		mi.sort()
		return sum(mi[:bn]) / float(bn)
		
		
	def get_sum_abs_alpha(self):
		return sum(map(abs, self.alpha_list))
	
	
	def get_derivative_gavg(self, ht, idx, y, bn, c):
		deriv_list = []
		for ix in idx:
			yi = self.y[ix]
			hti = ht[ix]
			l = self.get_argmax_ft(ix, yi)
			if hti == yi:
				'Scenario 1'
				res = c - (self.a_func_idx(ix, yi) - self.a_func_idx(ix, l))
				
			elif hti == l:
				'Scenario 2'
				res = -c - (self.a_func_idx(ix, yi) - self.a_func_idx(ix, l))
				
			else:
				'Scenario'
				res = -(self.a_func_idx(ix, yi) - self.a_func_idx(ix, l))
				
			deriv_list.append(res)
				
		return sum(deriv_list[:bn]) / float(bn)
	
	
	def line_search_max_margin(self, interval, th, n_worst_M, epsilon=1e-5):
		(beg, end), n_ = interval, n_worst_M
		c = self.get_sum_abs_alpha()
		bn = self.get_worst_M(n_)
		while (end-beg) > th:
			q = (beg + end) / 2.0
			deri_gavg = self.get_derivative_gavg(bn, c, q)
			if deri_gavg > 0:
				beg = q
			elif deri_gavg < 0:
				end = q
			else:
				"""
				when meet turning points, using
				``epsilon relaxation Algorithm``
				q + epsilon
				"""
				end = q + epsilon
				
		return (beg + end) / 2.0
		
			
	
	def get_argmax_margin(self, ht, idx, indice, X, y):
		"""
		because build tree we use min loss, though we need argmax margin.
		we return -max_argin
		
		parameters:
		-----------
		bn_rate: control size of bn'
		"""
		mx_margin = float(-sys.maxint)
		mx_l = -1
		bn = int(len(idx) * self.bn_rate)
		'we can optimal c here, using DP'
		c = self.get_sum_abs_alpha()
		for l in self.target_set:
			ht[idx[indice[y==l]]] = l 
			'calculate ft(xi, yi)'
			margin = self.get_derivative_gavg(self, ht, idx, bn, c)
			if margin > mx_margin:
				mx_margin = margin
				mx_l = l
		return mx_l, -mx_margin
		
	
	def margin_maximization(self):
		"""
		Phase II. margin Maximization Algorithm
		---------------------------------------
		
		somthing like svm max-soft-margin.
		In Phase I, we choose alpha using median of bst_interval.
		But In Phase II, we choose the optimalest alpha, it's a certain value.
		"""
		m, n = self.X.shape
		idx = np.arange(n)
		while self.t_ < self.max_round:	   
			ht = -np.ones(m, dtype=np.int)
			feature_list = copy.deepcopy(self.feature_list)
			ctree = tree(self.max_depth)
			self.build_tree(ctree, ht, idx, feature_list, 0, 1, self.get_argmax_margin)
			'when max margin, we need to append ht(x) first'
			self.ht_list.append(ht)
			bst_interval, bst_error = self.line_search(ht, self.X, self.y)
			alpha_t = self.line_search_max_margin(bst_interval, self.tol, int(n * self.bn_rate), self.eps)
			self.bst_error_list.append(bst_error)
			self.alpha_list.append(alpha_t)
			self.tree_list.append(ctree)
			self.t_ += 1
		
		
if __name__ == '__main__':
	params = {
		'max_initial_round':		50, 
		'max_margin_round':		   	10, 
		'bn_rate':				  	1,
		'max_depth':				2,
		'eps':					  	0.1,
		'tol':					  	0.01,
	}
	
	'make data'
	X, y = make_classification(
		n_samples=100,
		n_features=10,
		n_informative=7,
		n_redundant=1,
		n_classes=5,
		shuffle=True
	)
	X = X.astype(np.int)
	X_train, y_train = X[:16], y[:16]
	X_test, y_test = X[16:], y[16:]
	
	print 'X_train =', X_train
	print 'y_train =', y_train
	
	clf = DMCBoost(**params)
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	n_error = zero_one_loss(y_test, y_pred)
	error_rate = 1 - accuracy_score(y_test, y_pred)
	
	print 'y_test =', y_test
	print 'y_pred =', y_pred
	print 'n_error =', n_error, 'eror_rate =', error_rate
	