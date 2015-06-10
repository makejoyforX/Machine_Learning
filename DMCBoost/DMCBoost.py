#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import logging
import sys
from collections import defaultdict
from operator import itemgetter

eps = 1e-6

class DMCBoost(object):

	def __init__(self, max_initial_round, max_margin_round, n_worst_M, max_depth, tol=1e-3):
		self.max_initial_round = max_initial_round
		self.max_round = max_initial_round + max_margin_round
		self.max_depth = max_depth
		self.tol = tol
		
		
	def fit(self, X, y):
		self.X = X
		self.y = y
		m, n = X.shape
		self.target_set = np.unique(y)
		self.feature_list = map(
			lambda i: np.unique(X[:, i]), xrange(m)
		)
		
	
	def a_func_idx(idx, y):
		ret = 0.0
		for i,ht in enumerate(self.ht_list):
			if ht[idx]==y:
				ret += self.alpha_list[i] 
		return ret
	
	def get_l(self, idx, yi):
		"""
		 l = argmax_{y \in Y, y \neq yi} a(xi, y)
		"""
		mx_a = sys.minint
		mx_y = -1
		for y in self.target_Set:
			if y != yi:
				a_xi_y = self.a_func_idx(idx, y)
				if a_xi_y > mx_a:
					mx_a = a_xi_y
					mx_y = y
		return mx_y
				

	def line_search_min_loss(self, ht, idx, X, y):
		"""
		Algorithm 1.1
		
		parameters:
		------------
		ht:		h_t() function
		y:		y of the data
		l:		current label
		
		return:
		minError:	min Error
		interval:	optimal interval has the min Error
		
		"""
		e_dict = defaultdict(int)
		n = X.shape[0]
		for i in xrange(n):
			if ht[i] == y[i]:
				l = self.get_l(idx, y[i])
				q = a_func_idx(i, l) - a_func_idx(i, y[i])
				error_update = -1
			else:
				a_xi_yi = a_func_idx(i, y[i])
				a_xi_y = a_func_idx(i, ht[i]):
				if a_xi_yi > a_xi_y:
					q = a_xi_y - a_xi_yi
					error_update = 1
				else:
					continue
			e_dict[q] += error_update
		if len(e_dict) == 0:
			return (sys.minint, sys.maxint), 0.0
			
		intervals = sorted(e_dict.iteritems(), key=itemgetter(1))
		# incrementally calculate classification error
		mn_idx = -1
		minError = 0
		acc = 0
		length = len(intervals)
		intervals.append((-1, 0))
		for i in xrange(0, length):
			acc += intervals[i][1]
			if acc < mn:
				minError = acc
				mn_idx = i
				
		if mn_idx==-1:
			ret_interval = (sys.minint, intervals[0])
		elif mn_idx==length-1:
			ret_interval = (intervals[mn_idx], sys.maxint)
		else:
			ret_interval = (intervals[mn_idx], intervals[mn_idx+1])
			
		return ret_interval, minError

	
	def get_argmin_loss(self, ht, idx, indice, X, y):
		mn_loss = sys.maxint
		mn_l = -1
		for l in self.target_set:
			ht[idx[indice[y==l]]] = l
			loss = self.line_search_min_loss(ht[idx[indices]], idx, X, y)
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
				X_right, x_right = X[~indices], y[~indices]
				if len(X_left)==0 or len(X_right)==0:
					continue
				l_left , loss_left  = get_argmin_loss(ht, idx, indices, x_left, y_left)
				l_right, loss_right = get_argmin_loss(ht, idx, ~indices, x_right, y_right)
				if loss_left+loss_right < mn_loss:
					mn_loss = loss_left + loss_right
					bst_feat, bst_feat_val = feat, val
					bst_l_left, bst_l_right = l_left, l_right
		
		# add (bst_feat, bst_feat_val) to tree
		tr[rt] = (bst_feat, bst_feat_val)
		
		# delete bst_feat_val from bst_feat
		feature_list[bst_feat].remove(bst_feat_val)
		bst_indices = (X[:, bst_feat] == bst_feat_val)
		idx_left, idx_right = idx[bst_indices], idx[~bst_indices]
		ht[idx_left] = bst_l_left
		ht[idx_right] = bst_l_right
		
		self.build_tree(tr, ht, idx_left, feature_list, depth, rt<<1)
		self.build_tree(tr, ht, idx_right, feature_list, depth, rt<<1|1 )
		
	
	def initial_mode(self):
		"""
		Phase I. Minimizing Multi-class Classification Errors.
		
		"""
		self.t_ = 0
		self.ht_list = []
		self.bst_error_list = []
		self.alpha_list = []
		self.tree_list = []
		n, m = self.X.shape
		idx = np.arange(n)
		while self.t_ < self.max_initial_round:
			ht = np.zeros(m, )
			feature_list = list(self.feature_list)
			ctree = tree(self.max_depth)
			build_tree(ctree, ht, idx, feature_list, 0, 1)
			bst_interval, bst_error = self.line_search(ht, idx, self.X, self.y, self.get_argmin_loss)
			alpha_t = (bst_interval[0] + bst_interval[1]) / 2.0
			self.ht_list.append(ht)
			self.bst_error_list.append(bst_error)
			self.alpha_list.append(alpha_t)
			self.tree_list.append(tree)
			self.t_ += 1

			
	def ft_x_y(self, idx, y):
		return self.a_func_idx(idx, y)
	
	
	def get_argmax_ft(self, ix, yi):
		mx_ft_x_y = sys.minint
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
			ft_xi_yi = ft_x_y(ix, self.y[ix])
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
			yi = elf.y[ix]
			hti = ht[ix]:
			l = get_argmax_ft(ix, yi)
			if hti == yi:
				'Scenario 1'
				res = c - (a_func_idx(ix, yi) - a_func_idx(ix, l))
				
			elif hti == l:
				'Scenario 2'
				res = -c - (a_func_idx(ix, yi) - a_func_idx(ix, l))
				
			else:
				'Scenario'
				res = -(a_func_idx(ix, yi) - a_func_idx(ix, l))
				
			deriv_list.append(res)
				
		return sum(deriv_list[:bn]) / float(bn)
	
	def line_search_max_margin(self, interval, th, n_worst_M, epsilon=1e-5):
		beg, end, n_ = inerval, n_worst_M
		c = self.get_sum_abs_alpha()
		bn = self.get_worst_M(n_worst_M)
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
		worst_sample = 0.7
		"""
		mx_margin = sys.minint
		mx_l = -1
		worst_sample = 0.7
		bn = len(idx) * worst_sample
		'we can optimal c here, using DP'
		c = self.get_sum_abs_alpha()
		for l in self.target_set:
			ht[idx[indice[y==l]]] = l 
			'calculate ft(xi, yi)'
			margin = self.get_derivative_gavg(self, ht, idx, bn, c)
			if margin > mx_margin:
				mx_margin = margin
				mx_l = l
		return mx_l, -max_margin
		
	
	def margin_maximization(self):
		"""
		Phase II. margin Maximization Algorithm
		---------------------------------------
		
		somthing like svm max-soft-margin.
		In Phase I, we choose alpha using median of bst_interval.
		But In Phase II, we choose the optimalest alpha, it's a certain value.
		"""
		n, m = self.X.shape
		idx = np.arange(n)
		while self.t_ < self.max_round:	
			ht = np.zeros(m, )
			feature_list = list(self.feature_list)
			ctree = tree(self.max_depth)
			build_tree(ctree, ht, idx, feature_list, 0, 1, self.get_argmax_margin)
			'when max margin, we need to append ht(x) first'
			self.ht_list.append(ht)
			bst_interval, bst_error = self.line_search(ht, idx, self.X, self.y)
			alpha_t = self.line_search_max_margin(bst_interval, self.tol, self.n_worst_M)
			self.bst_error_list.append(bst_error)
			self.alpha_list.append(alpha_t)
			self.tree_list.append(tree)
			self.t_ += 1
		
		
if __name__ == '__main__':
	
