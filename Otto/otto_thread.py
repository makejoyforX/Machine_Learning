#!/usr/bin/env python

import os
import time
import pickle
import logging
import random
import threading


class otto_thread(threading.Thread):

	def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, verbose=None):
		threading.Thread.__init__(self, group=group, target=target, name=name, verbose=verbose)
		self.args = args
		self.kwargs = kwargs
		return 

	def run(self):
		if len(self.args)>0 and 'test' in self.kwargs and 'train' in self.kwargs:
			clf = self.args[0]
			logging.debug('threadName = %s, begin...' % (self.name))
			clf.fit(*self.kwargs['train'])
			logging.debug('%s: score is %.3f' % (self.name, clf.score(*self.kwargs['test'])))
			logging.debug('%s: bias is %.3f' % (self.name, clf.intercept_))
			logging.debug('threadName = %s, end...' % (self.name))
		return
