import numpy as np
import lmdb
import sys
sys.path.append("/home/axis-inside/Downloads/caffe-0.15.13/python")
sys.path.append("/home/axis-inside/Downloads/caffe-0.15.13/distribute/python")
sys.path.append("/usr/local/cuda-8.0/lib64")
import caffe
np.set_printoptions(threshold=np.nan)                                   ## To print all elements of an array

env = lmdb.open('/home/axis-inside/Audio_Data/anomaly_detection/40ms_same_bkgd_train', readonly=True)

with env.begin() as txn:
    raw_datum = txn.get(b'00000101')


datum = caffe.proto.caffe_pb2.Datum()
datum.ParseFromString(raw_datum)

flat_x = np.array(datum.float_data).astype(float)
x = flat_x.reshape(datum.channels, datum.height, datum.width)
y = datum.label
print x,y
##flat_x = np.fromstring(datum.float_data.extend, dtype=np.float32)
##x = flat_x.reshape(datum.channels, datum.height, datum.width)
##y = datum.label
##print x,y
##Iterating <key, value> pairs is also easy:

##with env.begin() as txn:
##    cursor = txn.cursor()
##    cursor.first()
##    key, value = cursor.item()
##    
##    #for key, value in cursor:
##    print(key, len(value))

