import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import time
from functools import partial
from tqdm import tqdm
from IPython.display import clear_output 
from data_prepare import *
from Network_structure import *
from loss_function import *
from train_method import *
from save_method import *
import sys
import os

# from __future__ import absolute_import
# from __future__ import print_function
# from __future__ import division

#sys.path.append('../')
from Novel_CNN import *
from BG_Attention import *
#from yhy_BG_Attention_v2 import *

# EEGdenoiseNet V2
# Author: Haoming Zhang 
# Here is the main part of the denoising neurl network, We can adjust all the parameter in the user-defined area.
#####################################################自定义 user-defined ########################################################

epochs = 50    # training epoch
batch_size  = 40    # training batch size
combin_num = 10    # combin EEG and noise ? times
denoise_network = 'BG_Attention'    # fcNN & Simple_CNN & Complex_CNN & RNN_lstm  & Novel_CNN & BiGRU_with_Attention
noise_type = 'EOG'


result_location = './'                     #  Where to export network results   ############ change it to your own location #########
foldername = 'EOG_bg_attention'            # the name of the target folder (should be change when we want to train a new network)
os.environ['CUDA_VISIBLE_DEVICES']='0'
save_train = False
save_vali = False
save_test = True


################################################## optimizer adjust parameter  ####################################################
rmsp=tf.optimizers.RMSprop(learning_rate=0.00005, rho=0.9)
adam=tf.optimizers.Adam(learning_rate=0.00005, beta_1=0.5, beta_2=0.9, epsilon=1e-08)
sgd=tf.optimizers.SGD(learning_rate=0.0002, momentum=0.9, decay=0.0, nesterov=False)

optimizer = rmsp

if noise_type == 'EOG':
  datanum = 512
elif noise_type == 'EMG':
  datanum = 1024

embedding = 16

# We have reserved an example of importing an existing network

#path = os.path.join(result_location, "EOG_bg_attention_embed16", "1" ,"denoised_model","saved_model.pb")
#denoiseNN = tf.keras.models.load_model(path)
#print("Done, loading model")
#################################################### 数据输入 Import data #####################################################

file_location = '../data/'                    ############ change it to your own location #########
if noise_type == 'EOG':
  EEG_all = np.load( file_location + 'EEG_all_epochs.npy')                              
  noise_all = np.load( file_location + 'EOG_all_epochs.npy') 
elif noise_type == 'EMG':
  EEG_all = np.load( file_location + 'EEG_all_epochs_512hz.npy')  
  noise_all = np.load( file_location + 'EMG_all_epochs_512hz.npy')

############################################################# Running #############################################################
#for i in range(10):
i = 1     # We run each NN for 10 times to increase  the  statistical  power  of  our  results
noiseEEG_train, EEG_train, noiseEEG_val, EEG_val, noiseEEG_test, EEG_test, test_std_VALUE = prepare_data(EEG_all = EEG_all, noise_all = noise_all, combin_num = 10, train_per = 0.8, noise_type = noise_type)


# print("train shape @@@@@@@@:" , noiseEEG_train.shape)

if denoise_network == 'fcNN':
  model = fcNN(datanum)

elif denoise_network == 'Simple_CNN':
  model = simple_CNN(datanum)

elif denoise_network == 'Complex_CNN':
  model = Complex_CNN(datanum)

elif denoise_network == 'RNN_lstm':
  model = RNN_lstm(datanum)

elif denoise_network == 'novel_cnn':
  model = Novel_CNN(datanum)

elif denoise_network == 'BG_Attention':
  model = BG_Attention(datanum, embedding)

elif denoise_network == 'yhy_BG_Attention':
  model = yhy_BG_Attention(datanum, embedding)

elif denoise_network == 'yhy_BG_Attention_v2':
  model = yhy_BG_Attention_v2(datanum, embedding)

else: 
  print('NN name arror')


saved_model, history = train(model, noiseEEG_train, EEG_train, noiseEEG_val, EEG_val, 
                      epochs, batch_size, optimizer, denoise_network, 
                      result_location, foldername , train_num = str(i))                        # steel the show   /   movie soul

denoised_test, test_mse = test_step_for_attention(saved_model, noiseEEG_test, EEG_test)

# save signal
save_eeg(saved_model, result_location, foldername, save_train, save_vali, save_test, 
                    noiseEEG_train, EEG_train, noiseEEG_val, EEG_val, noiseEEG_test, EEG_test, 
                    train_num = str(i))
np.save(result_location +'/'+ foldername + '/'+ str(i)  +'/'+ "nn_output" + '/'+ 'loss_history.npy', history)
np.save(result_location +'/'+ foldername + '/'+ str(i)  +'/'+ "nn_output" + '/'+ 'noise_value.npy', n)

#save model
#path = os.path.join(result_location, foldername, str(i+1), "denoise_model")
#tf.keras.models.save_model(saved_model, path)
