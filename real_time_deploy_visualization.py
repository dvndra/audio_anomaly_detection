import numpy as np
import matplotlib.pyplot as plt
import sys
import lmdb
import pyaudio
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
import copy
import pylab
import time
import copy
import cv2
from sklearn.externals import joblib
sys.path.append("/home/axis-inside/Downloads/caffe/python")
sys.path.append("/home/axis-inside/Downloads/caffe/distribute/python")
sys.path.append("/usr/local/cuda-8.0/lib64")
import caffe

frame = np.zeros((512,512,3), np.uint8)
############# 1. READ & ANALYZE AN AUDIO WINDOW ################################################################################################################################################################################
    
def audio_window(window_time, window_size, CHUNKSIZE, RATE):
    buffer_data = []                                                 ## declaring variable for buffering data
    moving_avg_data1 = np.zeros((20))                                ## Initialising array to store moving average of last 20 frames
    moving_avg_data2 = np.zeros((200))

    while True:

        numpydata = np.fromstring(stream.read(CHUNKSIZE), dtype=np.int16)              ## Convert audio dataframe into numpy data for further analysis

        ## Writing initial half of first window #################################################
        if window_time < window_size:
            buffer_data = copy.deepcopy(numpydata)

    ## Writing and analysing one window size with retention of half of the previous window ##
        elif window_time >= window_size:

            buffer_data = np.hstack((buffer_data, numpydata))           ## appending numpydata of latter half with intial half
            #window_time_amplitude= copy.deepcopy(buffer_data)
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

            ## Bark Frequency Binning of fft computed for a window ##############################
            barc_freq = [0, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500, 20500, 27000]
            bark = []
            bark, barc_freq = np.histogram(freq, bins = barc_freq,weights = abs(Z))
            bark = bark.reshape((26,1))

            ## MOVING AVERAGE OF LAST 20 FRAMES AMPLITUDE ACROSS ALL FREQUENCY BINS ##############
            moving_avg_data1 = np.delete(moving_avg_data1,0)
            moving_avg_data1 = np.append(moving_avg_data1,np.mean(bark))
            moving_avg1 = np.average(moving_avg_data1, weights = np.linspace(0.00, 0.1, num=20, endpoint=True, retstep=False, dtype=np.float32) )      # moving average plank/ladder-type
            moving_avg1 = 20*(np.log10(moving_avg1))                                                                                                    # moving average in decibel

            ## MOVING AVERAGE OF LAST 200 FRAMES AMPLITUDE ACROSS ALL FREQUENCY BINS ##############
            moving_avg_data2 = np.delete(moving_avg_data2,0)
            moving_avg_data2 = np.append(moving_avg_data2,np.mean(bark))
            moving_avg2 = np.average(moving_avg_data2, weights = np.linspace(0.00, 0.01, num=200, endpoint=True, retstep=False, dtype=np.float32) )    # moving average
            moving_avg2 = 20*(np.log10(moving_avg2))                                                                                                      # moving average in decibel

            #######################################################################################


            bark = 20*(np.log10(bark+0.000000001))                              # bark amplitude to decibel conversion
            bark1 = bark - moving_avg1                                          # moving average subtracted fft in decibel
            bark2 = bark - moving_avg2

            yield bark, bark1, bark2, window_time                               # yield all frequency amplitude
            buffer_data = copy.deepcopy(numpydata)                              # Updating last half of present window for next window

        window_time = window_time + (CHUNKSIZE/float(RATE))                     # increement of playtime to next chunk

###########################################################################################################################################################################################################
############ 2. FEATURE SCALING & MEAN-NORMALIZATION #######################################################################################################################################################

def feature_scaling_mean_normalization(data, train_mean, train_min, train_max):
    ####### (X-mu)/max(max-mu, mu-min) normalization is done. You can also opt to mu-sigma normalization ###########################################
    data[:,0:81:1]=(data[:,0:81:1]-train_mean[:,0:81:1])/(np.maximum((abs(train_min[:,0:81:1] - train_mean[:,0:81:1])),(train_max[:,0:81:1] - train_mean[:,0:81:1])))
    return data

#### Visualization ##############################################################################################################################
##    
def visualization(error, message):
##    t1=time.time()
    plt.figure()
    data = np.zeros((100))
    while (True):
        
        #pylab.plot(data)
        #pylab.title(message)
        #pylab.grid()
        #pylab.axis([0,len(data),-2**16/2,2**16/2])
        #pylab.savefig("03.png",dpi=150)
        #pylab.close('all')
        plt.plot(error)
        #plt.show(False)
        #plt.draw()
        
        
##    print("took %.02f ms"%((time.time()-t1)*1000))
        data = np.delete(data,0)
        data = np.append (data,error)
        yield 1
    
#####################################################################################################################################################
####### 4. MAIN FUNCTION ###########################################################################################################################

if __name__=="__main__":

##### Reading normalization parameters from training data pickle file ####
    train_mean, train_min, train_max = joblib.load('/home/axis-inside/Audio_Data/anomaly_detection/audio_files/40ms_3ft_demo_train_parameters.pkl')

    p=pyaudio.PyAudio()
    CHANNELS = 1
    RATE = 44100                                                              # sampling rate
    overlapping = 0.5
    window_size = 0.04                                                        # 40ms window size is used here
    CHUNKSIZE = int(RATE*window_size*overlapping)
    window_time = 0.00  
    data_combined = []
    print 'Recording Started' 
    ## window_time_amplitude = []
    p = pyaudio.PyAudio()                                                     ## create an audio object
    stream=p.open(format=pyaudio.paInt16,channels=1,rate=RATE,input=True,
                  frames_per_buffer=CHUNKSIZE)    
       
    window = audio_window(window_time, window_size, CHUNKSIZE, RATE)
    history0 = []                                                         ## array to store amplitude feature without mean subtraction
    history1 = []                                                         ## array to store first amplitude data frames for the duration
    history2 = []                                                         ## array to store second amplitude data frames for the duration
      
    bark, bark1, bark2, window_time = next (window)

    history0 = copy.deepcopy(bark)
    
    history1 = copy.deepcopy(bark1)
    
    history2= copy.deepcopy(bark2)
    
    num_frames_append = 50
    for i in range (1,num_frames_append):
         bark, bark1, bark2, window_time = next (window)
         history0 = np.hstack((history0,bark))
         history1 = np.hstack((history1,bark1))
         history2 = np.hstack((history2,bark2))
    history0 = np.vstack((history0,np.sum(history0,axis=0)))
    history1 = np.vstack((history1,np.sum(history1,axis=0)))
    history2 = np.vstack((history2,np.sum(history2,axis=0)))
    data_combined = np.vstack((history0,history1,history2))
    
    
    #### feature scaling & mean normalization ###############################
    data_combined = np.transpose(data_combined)                             ## convert to num_samples x features
    data_combined = feature_scaling_mean_normalization(data_combined, train_mean, train_min, train_max)
   
    ##### convert into NCHW format as compatible to caffe ####################
    data_combined = data_combined[:,:,newaxis,newaxis]                      ## 2D to 4D array
    data_combined= np.transpose(data_combined,(2,3,0,1))

    ## FORWARD PASS #######
    model = '/home/axis-inside/Downloads/caffe/examples/anomaly_40ms_same_bkgd_pic_3features/deploy.prototxt';
    weights = '/home/axis-inside/Downloads/caffe/examples/anomaly_40ms_same_bkgd_pic_3features/trained_model_demo_iter_40.caffemodel';
    caffe.set_mode_gpu();
    caffe.set_device(0);
    net = caffe.Net(model, weights, caffe.TEST);
    net.blobs['data'].reshape(*data_combined.shape)
    net.blobs['data'].data[...] = data_combined
    res = net.forward()
    print res, window_time
    
    ##### cut - off for anomaly ################################################
    cut_off = 100
    if((res['l2_error']) > cut_off):
        message = 'anomaly_detected'      
    else:
        message = 'normal'
    print message
    #plot = visualization((res['l2_error']),message)
    #z = next (plot)
    ######## READING DATA TILL THE END ####################################################################    
    while (True):
        frame = np.zeros((512,512,3), np.uint8)
        bark, bark1, bark2, window_time = next (window)
        bark = np.vstack((bark,np.sum(bark,axis=0)))
        bark1 = np.vstack((bark1,np.sum(bark1,axis=0)))
        bark2 = np.vstack((bark2,np.sum(bark2,axis=0)))
        
        ##### First in last out to maintain 50 windows in a frame sample #########
        history0 = np.delete(history0,0,1)                             
        history0 = np.hstack((history0,bark))
        history1 = np.delete(history1,0,1) 
        history1 = np.hstack((history1,bark1))
        history2 = np.delete(history2,0,1) 
        history2= np.hstack((history2,bark2))
        data_combined = np.vstack((history0,history1,history2))
        
        ##### feature scaling & mean normalization ###############################
        data_combined = np.transpose(data_combined)                             ## convert to num_samples x features
        data_combined = feature_scaling_mean_normalization(data_combined, train_mean, train_min, train_max)
        
        ##### convert into NCHW format as compatible to caffe ####################
        data_combined = data_combined[:,:,newaxis,newaxis]                      ## 2D to 4D array
        data_combined= np.transpose(data_combined,(2,3,0,1))

        ##### do forward pass ####################################################
        net.blobs['data'].data[...] = data_combined
        res = net.forward()                                                     ## net.forward(start='layer1') is another way to omit the data layer computation again in forward apart from remove it in prototxt ##
        print res, window_time        
    
        ######### cut - off for anomaly #######################################################################################################
        cut_off = 100
        if((res['l2_error']) > cut_off):
            message = 'anomaly'
            frame[:,:,2]= 255
        else:
            message = 'normal'
            frame[:,:,1]= 255
        print message
        #z = next (plot)
        cv2.putText(frame, message,(125,256), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255),2)
        cv2.imshow('Audio_analytics', frame)
        cv2.waitKey(1)

    ###########################################################################################################################################
    # CLEANUP STUFF
    cv2.destroyAllWindows
    stream.stop_stream()
    stream.close()
    p.terminate()

## Visualization
    
##def soundplot(stream):
##    t1=time.time()
##    for j in range(0,1):
##        data = np.fromstring(stream.read(CHUNK),dtype=np.int16)
##        print np.shape (data)
##        history = copy.deepcopy(data)
##    data = np.fromstring(stream.read(CHUNK),dtype=np.int16)
##    history= np.hstack((history,data))
##    print np.shape(history)
##    pylab.plot(history)
##    pylab.title(i)
##    pylab.grid()
##    pylab.axis([0,len(history),-2**16/2,2**16/2])
##    pylab.savefig("03.png",dpi=150)
##    pylab.close('all')
##    print("took %.02f ms"%((time.time()-t1)*1000))
##    history = copy.deepcopy(data)




