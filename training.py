import pyaudio
import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt
from scipy import signal
from sklearn import mixture
import pickle
import wave
from numpy import newaxis
import contextlib
import os
import re
import math
import lmdb
import sys
import copy
from sklearn.externals import joblib
sys.path.append("/home/axis-inside/Downloads/caffe-0.15.13/python")
sys.path.append("/home/axis-inside/Downloads/caffe-0.15.13/distribute/python")
sys.path.append("/usr/local/cuda-8.0/lib64")
import caffe

np.set_printoptions(threshold=np.nan)                                   ## To print all elements of an array

os.getcwd()                                                             ## Get the current directory
os.chdir('/home/axis-inside/Audio_Data/anomaly_detection/audio_files/demo_train')                 ## change the directory
training_list = os.listdir('.')                                         ## list all the files in the current directory

## MAIN Function
data=[]

for file in training_list:

    def audio_window(window_time, window_size, CHUNKSIZE,RATE,file):

        buffer_data = []                                                 ## declaring variable for buffering data
        wf = wave.open(file, 'rb')
        moving_avg_data1 = np.zeros((20))                                ## Initialising array to store moving average of last 20 frames
        moving_avg_data2 = np.zeros((200))

        while True:

            data = wf.readframes(CHUNKSIZE)                              ## Read & stream data based on the chunk size
            numpydata = np.fromstring(data, dtype=np.int16)              ## Convert audio dataframe into numpy data for further analysis

            ## Writing initial half of first window ###
            if window_time < window_size:
                buffer_data = copy.deepcopy(numpydata)

            ## Writing and analysing one window size with retention of half of the previous window ##
            elif window_time>= window_size:

                buffer_data = np.hstack((buffer_data, numpydata))           ## appending numpydata of latter half to already available buffer_data for intial half

                Z = []                                                      ## declaring variable for computing DFT of single window
                n = len(buffer_data)                                        ## length of the signal of a window
                w = np.hanning(n)                                           ## applying hanning window to avoid aliasing
                buffer_data = w*buffer_data
                k = np.arange(n)
                T = (n*1.0)/RATE
                frq = k/T                                                   # frequency range till sampling RATE (Nyquist frequency)
                freq = frq[range(int(n/2))]                                 # frequency range till half of sampling rate
                Z = np.fft.fft(buffer_data)/n                               # fft computing and normalization
                Z = Z[range(int(n/2))]

                ## Bark Frequency Binning of fft computed for a window ##
                barc_freq = [0, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500, 20500, 27000]
                bark = []
                bark, barc_freq = np.histogram(freq, bins = barc_freq,weights = abs(Z))
                bark = bark.reshape((26,1))

                ## MOVING AVERAGE OF LAST 20 FRAMES AMPLITUDE ACROSS ALL FREQUENCY BINS #################################
                moving_avg_data1 = np.delete(moving_avg_data1,0)
                moving_avg_data1 = np.append(moving_avg_data1,np.mean(bark))
                moving_avg1 = np.average(moving_avg_data1, weights = np.linspace(0.00, 0.1, num=20, endpoint=True, retstep=False, dtype=np.float32) )      # moving average plank/ladder-type
                moving_avg1 = 20*(np.log10(moving_avg1))                                                                                                    # moving average in decibel

                ## MOVING AVERAGE OF LAST 200 FRAMES AMPLITUDE ACROSS ALL FREQUENCY BINS #################################
                moving_avg_data2 = np.delete(moving_avg_data2,0)
                moving_avg_data2 = np.append(moving_avg_data2,np.mean(bark))
                moving_avg2 = np.average(moving_avg_data2, weights = np.linspace(0.00, 0.01, num=200, endpoint=True, retstep=False, dtype=np.float32) )    # moving average
                moving_avg2 = 20*(np.log10(moving_avg2))                                                                                                      # moving average in decibel

                ######################################################################################################


                bark = 20*(np.log10(bark+0.000000001))                                          # bark amplitude to decibel conversion
                bark1 = bark - moving_avg1                                          # moving average subtracted fft in decibel
                bark2 = bark - moving_avg2

                yield bark, bark1, bark2, freq, window_time                         # yield all frequency amplitude
                buffer_data = copy.deepcopy(numpydata)                                             ## Updating last half of present window for next window

            window_time = window_time + (CHUNKSIZE/float(RATE))                     ## increement of playtime to next chunk
    
    wf = wave.open(file, 'rb')
    p = pyaudio.PyAudio()                                                     ## create an audio object

    stream = p.open(format = p.get_format_from_width(wf.getsampwidth()),
                    channels = wf.getnchannels(),
                    rate = wf.getframerate(),
                    output = True)                                          # open stream based on the wave object which has been input
    CHANNELS = 1
    RATE = 44100                                                            # sampling rate
    overlapping = 0.5
    window_size = 0.04                                                       # 100ms window size is used here
    CHUNKSIZE = int(RATE*window_size*overlapping)
    window_time = 0.00
    print 'Number of channels in input audio = %d' %wf.getnchannels()
    print 'Sampling rate = %d' %wf.getframerate()

    ### Computing duration of wave file ###
    with contextlib.closing (wf) as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        duration = frames / float(rate)
        print 'Audio duration = %.3f' %duration
    print "Analyzing music file %s" %file

    window = audio_window(window_time, window_size, CHUNKSIZE,RATE, file)
    history0 = []                             ## array to store amplitude feature without mean subtraction
   
    history1 = []                                                         ## array to store first amplitude data frames for the duration
    history2 = []                                                         ## array to store second amplitude data frames for the duration
    

   
    bark, bark1, bark2, freq, window_time = next (window)

    history0 = copy.deepcopy(bark)
    
    history1 = copy.deepcopy(bark1)
    
    history2= copy.deepcopy(bark2)
        
    temp=0
    while (window_time < duration):
        ##print ('window_time=',window_time, ', duration=', duration)
        bark, bark1, bark2, freq, window_time = next (window)
        ## Bark Frequency Binning of moving average subtracted fft computed for a window ##

        history0 = np.hstack((history0,bark))
        
        history1 = np.hstack((history1,bark1))
        
        history2= np.hstack((history2,bark2))
               
        if (round(window_time/duration*100,1))%10==0:
            if temp != round(window_time/duration*100,1):
                print 'Finished '+repr(round(window_time/duration*100,1))+' %'
            temp = round(window_time/duration*100,1)

    history0 = np.vstack((history0,np.sum(history0,axis=0)))
    history1 = np.vstack((history1,np.sum(history1,axis=0)))
    history2 = np.vstack((history2,np.sum(history2,axis=0)))

    
    all_combined = np.vstack((history0,history1,history2))
    data.append(all_combined)
    
    # CLEANUP STUFF
    stream.close()
    p.terminate()

##print np.shape(data)
data= np.hstack(data[:])
data= data.astype(np.float32)
print np.shape(data)
data= np.transpose(data)                    # converting into format samples x features
print np.shape(data)

## Feature Scaling & Mean Normalization

pic_data_mean = np.mean(data[:,0:81:1],axis=0)
pic_data_min = np.min(data[:,0:81:1],axis=0)
pic_data_max = np.max(data[:,0:81:1],axis=0)
##print np.shape(pic_data_min), np.shape(pic_data_mean), pic_data_max
pic_data_mean = np.transpose(pic_data_mean.reshape((81,1)))
pic_data_min = np.transpose(pic_data_min.reshape((81,1)))
pic_data_max = np.transpose(pic_data_max.reshape((81,1)))
##print np.shape(pic_data_min), np.shape(pic_data_mean), pic_data_max

data[:,0:26:1]=(data[:,0:26:1]-pic_data_mean[:,0:26:1])/(np.maximum((abs(pic_data_min[:,0:26:1] - pic_data_mean[:,0:26:1])),(pic_data_max[:,0:26:1] - pic_data_mean[:,0:26:1])))
data[:,27:53:1]=(data[:,27:53:1]-pic_data_mean[:,27:53:1])/(np.maximum((abs(pic_data_min[:,27:53:1] - pic_data_mean[:,27:53:1])),(pic_data_max[:,27:53:1] - pic_data_mean[:,27:53:1])))
data[:,54:80:1]=(data[:,54:80:1]-pic_data_mean[:,54:80:1])/(np.maximum((abs(pic_data_min[:,54:80:1] - pic_data_mean[:,54:80:1])),(pic_data_max[:,54:80:1] - pic_data_mean[:,54:80:1])))

data[:,26]=(data[:,26]-np.mean(data[:,26]))/(max((abs(np.min(data[:,26]) - np.mean(data[:,26])),(np.max(data[:,26])) - (np.mean(data[:,26])))))
data[:,53]=(data[:,53]-np.mean(data[:,53]))/(max((abs(np.min(data[:,53]) - np.mean(data[:,53])),(np.max(data[:,53])) - (np.mean(data[:,53])))))
data[:,80]=(data[:,80]-np.mean(data[:,80]))/(max((abs(np.min(data[:,80]) - np.mean(data[:,80])),(np.max(data[:,80])) - (np.mean(data[:,80])))))


print np.min(data), np.max(data)
print np.shape (data)
data = data[:,:,newaxis]                    # 2-D to 3-D array
total_frames, total_features, channels = np.shape(data)


# Converting data into picture format for caffe #

num_frames_append = 50

pic_data_array = np.ones((num_frames_append,total_features,(total_frames + 1 - num_frames_append) ))

for k in range(0, (total_frames + 1 - num_frames_append)):
    
    for j in range(0,total_features):
        
        for i in range(0,num_frames_append):
            pic_data_array[i,j,k]= data[(i+k),j,0] 
    
print np.shape(pic_data_array)

pic_data_array = pic_data_array[:,:,:,newaxis]                    # 3-D to 4-D array
pic_data_array= np.transpose(pic_data_array,(2,3,0,1))
print np.shape(pic_data_array)
##samples, channels, height, width = np.shape(pic_data.array)

##### Dumping variables ######
os.chdir('..')
joblib.dump(pic_data_array, '40ms_3ft_demo_train.pkl')
joblib.dump([pic_data_mean, pic_data_min, pic_data_max], '40ms_3ft_demo_train_parameters.pkl')


### Converting numpy array to lmdb ######
# Let's pretend this is interesting data
##X = joblib.load('pic_data.pkl')
##y = np.zeros(samples, dtype=np.int64)
##
### We need to prepare the database for the size. We'll set it 10 times
### greater than what we theoretically need. There is little drawback to
### setting this too big. If you still run into problem after raising
### this, you might want to try saving fewer entries in a single
### transaction.
##map_size = 429496729600
##
##env = lmdb.open('40ms_same_bkgd_train', map_size=map_size)
##
##with env.begin(write=True) as txn:
##    # txn is a Transaction object
##    for i in range(samples):
##        datum = caffe.proto.caffe_pb2.Datum()
##        datum.channels = X.shape[1]
##        datum.height = X.shape[2]
##        datum.width = X.shape[3]
##        print i
##        datum.float_data.extend(X[i].astype(float).flat)
##        datum.label = int(y[i])
##        str_id = '{:08}'.format(i)
##        txn.put(str_id.encode('ascii'), datum.SerializeToString())
####




