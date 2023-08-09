import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# BG-Attention
Denoiseoutput_BG = np.load('./code/yh/Denoiseoutput_test.npy', allow_pickle=True)
EEG_test_BG = np.load('./code/yh/EEG_test.npy', allow_pickle=True)
Denoiseoutput_BG = Denoiseoutput_BG.squeeze()

# Novel CNN
Denoiseoutput_CNN = np.load('./code/Novel_CNN_EOG/Denoiseoutput_test.npy', allow_pickle=True)
EEG_test_CNN = np.load('./code/Novel_CNN_EOG/EEG_test.npy', allow_pickle=True)

# RNN
Denoiseoutput_RNN = np.load('./code/EOG_LSTM/Denoiseoutput_test.npy', allow_pickle=True)
EEG_test_RNN = np.load('./code/EOG_LSTM/EEG_test.npy', allow_pickle=True)


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


x_axis = []

for i in range(0, len(Denoiseoutput_BG[0])):                          # x축 설정
    x_axis = np.append(x_axis, i)


RRMSE_index = []
RRMSE_bg = []
RRMSE_cnn = []
RRMSE_rnn = []


for i in range(1, 11, 3):                                          # 논문의 내용과 같이 (-7, -4, -1, 2)dB 값을 4번 출력
    RRMSE_BG = []
    RRMSE_CNN = []
    RRMSE_RNN = []
    
    RRMSE_index.append(i - 8)
    
    for j in range((i - 1) * 340, i * 340):
        denoiseoutput_BG = Denoiseoutput_BG[j]
        eeg_test_BG = EEG_test_BG[j]

        denoiseoutput_CNN = Denoiseoutput_CNN[j]
        eeg_test_CNN = EEG_test_CNN[j]

        denoiseoutput_RNN = Denoiseoutput_RNN[j]
        eeg_test_RNN = EEG_test_RNN[j]
    
        RRMSE_BG.append(denoise_loss_mse(denoiseoutput_BG, eeg_test_BG).numpy())
        RRMSE_CNN.append(denoise_loss_mse(denoiseoutput_CNN, eeg_test_CNN).numpy())
        RRMSE_RNN.append(denoise_loss_mse(denoiseoutput_RNN, eeg_test_RNN).numpy())
        
    RRMSE_av_bg = sum(RRMSE_BG) / 340
    RRMSE_av_cnn = sum(RRMSE_CNN) / 340
    RRMSE_av_rnn = sum(RRMSE_RNN) / 340

    RRMSE_bg.append(RRMSE_av_bg)
    RRMSE_cnn.append(RRMSE_av_cnn)
    RRMSE_rnn.append(RRMSE_av_rnn)

df = pd.DataFrame({'BG-Attention' : RRMSE_bg, 'Novel CNN' : RRMSE_cnn, 'LSTM' : RRMSE_rnn}, index = RRMSE_index)
print(df)
print('RRMSE_bg:', RRMSE_bg, 'RRMSE_cnn:', RRMSE_cnn, 'RRMSE_rnn:', RRMSE_rnn)

fig, ax = plt.subplots(figsize=(12,6))
bar_width = 0.25

index = np.arange(4)

b1 = plt.bar(index, df['BG-Attention'], bar_width, alpha=0.4, color='red', label='BG-Attention')
b2 = plt.bar(index + bar_width, df['Novel CNN'], bar_width, alpha=0.4, color='blue', label='Novel CNN')
b3 = plt.bar(index + 2 * bar_width, df['LSTM'], bar_width, alpha=0.4, color='green', label='LSTM')

plt.xticks(np.arange(bar_width, 4 + bar_width, 1), RRMSE_index)

plt.xlabel('SNR /dB', size = 13)
plt.ylabel('MSE', size = 13)
plt.legend()
plt.show()

#plt.savefig(f'RRMSE_per_Model')  # png 저장
