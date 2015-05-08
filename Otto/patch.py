#!/usr/bin/env python
import os
import sys

def otto_patch(fname):
	desName = 'zyx_otto_sub.csv'
	header = 'id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9\n'
	with open(fname, 'r') as fin:
		lines = fin.readlines()
		with open(desName, 'w') as fout:
			fout.write(header)
			for i,line in enumerate(lines):
				fout.write('%d,%s' % (i+1,line))

def otto_patch2(fname):
	desName = 'zyx_otto_sub.csv'
	header = 'id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9\n'
	with open(fname, 'r') as fin:
		lines = fin.readlines()[1:]
		with open(desName, 'w') as fout:
			fout.write(header)
			print lines[0].count(',')
			for i,line in enumerate(lines):
				p = line.find(',')
				fout.write('%d%s' % (i+1,line[p:]))



if __name__ == '__main__':
	if len(sys.argv)>1:
		fname = sys.argv[1]
	else:
		fname = 'zyx_otto_submission_133.csv'
	otto_patch(fname)