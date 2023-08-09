import tensorflow as tf
# from loss_function import *
import numpy as np


Denoiseoutput = np.load('./code/yh/Denoiseoutput_test.npy', allow_pickle=True)
EEG_test = np.load('./code/yh/EEG_test.npy', allow_pickle=True)
Denoiseoutput = Denoiseoutput.squeeze()

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
CC_dB = []
CC_std = []

for i in range(1, 11, 3):                                          # 논문의 내용과 같이 (-7, -4, -1, 2)dB 값을 4번 출력
    CC = []
    minimum = []
    CC_index.append(i - 8)
    
    for j in range((i - 1) * 340, i * 340):
        denoiseoutput = Denoiseoutput[j]
        eeg_test = EEG_test[j]
    
        CC.append(calculate_correlation_coefficient(eeg_test, denoiseoutput))
        
    CC_Value = sum(CC) / 340
    CC_s = np.std(CC)

    CC_dB.append(CC_Value)
    CC_std.append(CC_s)

print('CC_mean:', CC_dB, 'CC_Standard_Deviation:', CC_std, 'index:', CC_index)
