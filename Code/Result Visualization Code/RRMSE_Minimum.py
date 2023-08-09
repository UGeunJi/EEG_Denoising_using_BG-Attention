import tensorflow as tf
# from loss_function import *
import numpy as np


Denoiseoutput = np.load('./code/yh/Denoiseoutput_test.npy', allow_pickle=True)
EEG_test = np.load('./code/yh/EEG_test.npy', allow_pickle=True)
Denoiseoutput = Denoiseoutput.squeeze()

################################################################## function #################################################################

def denoise_loss_mse(denoise, clean):      
  loss = tf.losses.mean_squared_error(denoise, clean)
  return tf.reduce_mean(loss)

def denoise_loss_rmse(denoise, clean):      #tmse
  loss = tf.losses.mean_squared_error(denoise, clean)
  #loss2 = tf.losses.mean_squared_error(noise, clean)
  return tf.math.sqrt(tf.reduce_mean(loss))

def denoise_loss_rrmset(denoise, clean):      #tmse
  rmse1 = denoise_loss_rmse(denoise, clean)
  rmse2 = denoise_loss_rmse(clean, tf.zeros(clean.shape[0], tf.float64))
  #print(f'######################################## {rmse1}, {rmse2}')
  #loss2 = tf.losses.mean_squared_error(noise, clean)
  return rmse1/rmse2

################################################################## function #################################################################

RRMSE_index = []
RRMSE_min = []

for i in range(1, 11, 3):                                          # 논문의 내용과 같이 (-7, -4, -1, 2)dB 값을 4번 출력
    minimum = []
    RRMSE_index.append(i - 8)
    
    for j in range((i - 1) * 340, i * 340):
        denoiseoutput = Denoiseoutput[j]
        eeg_test = EEG_test[j]
    
        minimum.append(denoise_loss_rrmset(denoiseoutput, eeg_test).numpy())
        
    minimum_value = min(minimum)

    RRMSE_min.append(minimum_value)

print('RRMSE_min:', RRMSE_min, 'RRMSE_index:', RRMSE_index)        
