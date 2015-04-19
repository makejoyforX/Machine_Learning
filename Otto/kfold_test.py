#!/usr/bin/env python

import numpy as np
from sklearn.cross_validation import KFold

X = np.arange(40)
X = X.reshape(20, 2)
y = np.arange(20)

kf = KFold(20, n_folds=10, shuffle=True)
print kf

for train_index, test_index in kf:
	print 'Train:', train_index
	print 'Test:', test_index
