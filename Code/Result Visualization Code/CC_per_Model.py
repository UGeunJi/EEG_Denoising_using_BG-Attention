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

def calculate_correlation_coefficient(array1, array2):
    # Ensure both arrays have the same length
    if len(array1) != len(array2):
        raise ValueError("Arrays must have the same length.")

    # Calculate the correlation coefficient
    correlation_coefficient = np.corrcoef(array1, array2)[0, 1]

    return correlation_coefficient

################################################################## function #################################################################


CC_index = []
CC_bg = []
CC_cnn = []
CC_rnn = []


for i in range(1, 11, 3):                                          # 논문의 내용과 같이 (-7, -4, -1, 2)dB 값을 4번 출력
    CC_BG = []
    CC_CNN = []
    CC_RNN = []
    
    CC_index.append(i - 8)
    
    for j in range((i - 1) * 340, i * 340):
        denoiseoutput_BG = Denoiseoutput_BG[j]
        eeg_test_BG = EEG_test_BG[j]

        denoiseoutput_CNN = Denoiseoutput_CNN[j]
        eeg_test_CNN = EEG_test_CNN[j]

        denoiseoutput_RNN = Denoiseoutput_RNN[j]
        eeg_test_RNN = EEG_test_RNN[j]
    
        CC_BG.append(calculate_correlation_coefficient(denoiseoutput_BG, eeg_test_BG))
        CC_CNN.append(calculate_correlation_coefficient(denoiseoutput_CNN, eeg_test_CNN))
        CC_RNN.append(calculate_correlation_coefficient(denoiseoutput_RNN, eeg_test_RNN))
        
    CC_av_bg = sum(CC_BG) / 340
    CC_av_cnn = sum(CC_CNN) / 340
    CC_av_rnn = sum(CC_RNN) / 340

    CC_bg.append(CC_av_bg)
    CC_cnn.append(CC_av_cnn)
    CC_rnn.append(CC_av_rnn)

################################################################### Line Plot #############################################################################

print('CC_bg:', CC_bg, 'CC_cnn:', CC_cnn, 'CC_rnn:', CC_rnn)

plt.title(f'CC Average per Model')
plt.plot(CC_index, CC_bg, linestyle='-', label='BG-Attention')
plt.plot(CC_index, CC_cnn, linestyle='-', label='Novel CNN')
plt.plot(CC_index, CC_rnn, linestyle='-', label='LSTM')
plt.xlabel('SNR /dB')
plt.ylabel('CC')
plt.legend()
plt.show()

#plt.savefig(f'CC_per_Model')  # png 저장


################################################################### Bar Plot #############################################################################


'''
df = pd.DataFrame({'BG-Attention' : CC_bg, 'Novel CNN' : CC_cnn, 'LSTM' : CC_rnn}, index = CC_index)


fig, ax = plt.subplots(figsize=(12,6))
bar_width = 0.25

index = np.arange(4)

b1 = plt.bar(index, df['BG-Attention'], bar_width, alpha=0.4, color='red', label='BG-Attention')
b2 = plt.bar(index + bar_width, df['Novel CNN'], bar_width, alpha=0.4, color='blue', label='Novel CNN')
b3 = plt.bar(index + 2 * bar_width, df['LSTM'], bar_width, alpha=0.4, color='green', label='LSTM')

plt.xticks(np.arange(bar_width, 4 + bar_width, 1), CC_index)

plt.xlabel('CC', size = 13)
plt.ylabel('SNR /dB', size = 13)
plt.legend()
plt.show()

#plt.savefig(f'CC_per_Model')  # png 저장
'''
