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

def calculate_snr(signal, noise):
    # Calculate the mean of the signal and noise arrays
    signal_mean = np.mean(signal)
    noise_mean = np.mean(noise)
    
    # Calculate the sum of squared differences between the signal and its mean
    signal_power = np.sum((signal - signal_mean) ** 2)
    
    # Calculate the sum of squared differences between the noise and its mean
    noise_power = np.sum((noise - noise_mean) ** 2)
    
    # Calculate the SNR
    snr = 10 * np.log10(signal_power / noise_power)
    
    return snr

################################################################## function #################################################################

'''
x_axis = []

for i in range(0, len(Denoiseoutput_BG[0])):                          # x축 설정
    x_axis = np.append(x_axis, i)
'''

Paper = []
ex_result = []
bg_paper = []

RRMSE_bg = []
RRMSE_cnn = []
RRMSE_rnn = []

x_axis = ['-7', '-4', '-1', '2']
for i in range(1, 11, 3):                                          # 논문의 내용과 같이 (-7, -4, -1, 2)dB 값을 4번 출력
    RRMSE_BG = []
    RRMSE_CNN = []
    RRMSE_RNN = []
    
    for j in range((i - 1) * 340, i * 340):
        denoiseoutput_BG = Denoiseoutput_BG[j]
        eeg_test_BG = Denoiseoutput_BG[j] - EEG_test_BG[j]

        denoiseoutput_CNN = Denoiseoutput_CNN[j]
        eeg_test_CNN = Denoiseoutput_CNN[j] - EEG_test_CNN[j]

        denoiseoutput_RNN = Denoiseoutput_RNN[j]
        eeg_test_RNN = Denoiseoutput_RNN[j] - EEG_test_RNN[j]
    
        RRMSE_BG.append(calculate_snr(denoiseoutput_BG, eeg_test_BG))
        RRMSE_CNN.append(calculate_snr(denoiseoutput_CNN, eeg_test_CNN))
        RRMSE_RNN.append(calculate_snr(denoiseoutput_RNN, eeg_test_RNN))
        
    RRMSE_av_bg = sum(RRMSE_BG) / 340
    RRMSE_av_cnn = sum(RRMSE_CNN) / 340
    RRMSE_av_rnn = sum(RRMSE_RNN) / 340

    RRMSE_bg.append(RRMSE_av_bg)
    RRMSE_cnn.append(RRMSE_av_cnn)
    RRMSE_rnn.append(RRMSE_av_rnn)

df = pd.DataFrame({'FCNN' : RRMSE_cnn, 'BG-Attention' : RRMSE_bg, 'RNN' : RRMSE_rnn}, index = x_axis)

#print('Paper:', Paper, 'ex_result:', ex_result, 'bg_paper:', bg_paper)

fig, ax = plt.subplots(figsize=(12,6))
bar_width = 0.25

index = np.arange(4)

b1 = plt.bar(index, df['FCNN'], bar_width, alpha=0.4, color='red', label='FCNN')
b2 = plt.bar(index + bar_width, df['BG-Attention'], bar_width, alpha=0.4, color='blue', label='BG-Attention')
b3 = plt.bar(index + 2 * bar_width, df['RNN'], bar_width, alpha=0.4, color='green', label='RNN')

plt.xticks(np.arange(bar_width, 4 + bar_width, 1), x_axis)

plt.xlabel('SNR/dB', size = 13)
plt.ylabel('SNR /dB', size = 13)
plt.legend()
plt.show()

#plt.savefig(f'RRMSE_per_Model')  # png 저장
