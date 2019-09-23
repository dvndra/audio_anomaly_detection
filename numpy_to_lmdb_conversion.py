# Converting numpy array to lmdb ######
import pyaudio
import numpy as np
import pickle
import contextlib
import os
import math
import lmdb
import sys
sys.path.append("/home/axis-inside/Downloads/caffe-0.15.13/python")
sys.path.append("/home/axis-inside/Downloads/caffe-0.15.13/distribute/python")
sys.path.append("/usr/local/cuda-8.0/lib64")
import caffe
from sklearn.externals import joblib

# Let's pretend this is interesting data

X = joblib.load('/home/axis-inside/Audio_Data/anomaly_detection/audio_files/40ms_3ft_demo_train.pkl')

print np.shape(X)
samples, channels, height, width = np.shape(X)
y = np.zeros(samples, dtype=np.int64)

# We need to prepare the database for the size. We'll set it 10 times
# greater than what we theoretically need. There is little drawback to
# setting this too big. If you still run into problem after raising
# this, you might want to try saving fewer entries in a single
# transaction.
map_size = 429496729600

env = lmdb.open('40ms_3ft_demo_train', map_size=map_size)

with env.begin(write=True) as txn:
    # txn is a Transaction object
    for i in range(samples):
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = X.shape[1]
        datum.height = X.shape[2]
        datum.width = X.shape[3]
        print i
        datum.float_data.extend(X[i].astype(float).flat)
        datum.label = int(y[i])
        str_id = '{:08}'.format(i)
        txn.put(str_id.encode('ascii'), datum.SerializeToString())
##
