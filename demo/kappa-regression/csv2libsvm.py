#!/usr/bin/python
import sys
import csv

if len(sys.argv) < 3:
    print 'Usage: <csv> <libsvm> <label_index> <skip_header>'
    print 'convert csv to libsvm'

try:
	label_index = int( sys.argv[3] )
except IndexError:
	label_index = 0

try:
	skip_header = sys.argv[4]
except IndexError:
	skip_header = 0

fo = open(sys.argv[2], 'w')

fi = open(sys.argv[1])
reader = csv.reader( fi )

if skip_header:
	header = reader.next()

for line in reader:
	if label_index < -1:
		label = 1
	else:
		if label_index == -1:
			label_index = len(line)-1
		label = line.pop( label_index )
	fo.write('%s' % label)
	for i in xrange(len(line)):
		try:
			if line[i]=='' or float( line[i] )==0.0:
				continue
			fo.write(' %d:%s' % (i, line[i]))
		except:
			fo.write(' %d:"%s"' % (i, line[i]))
	fo.write('\n')

fo.close()
fi.close()
