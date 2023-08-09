import numpy as np
import os
import matplotlib.pyplot as plt
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

file_name = 'Novel_CNN_EOG'

noiseinput = np.load('./code/' + file_name + '/EEG_test.npy', allow_pickle=True)
Denoiseoutput = np.load('./code/' + file_name + '/Denoiseoutput_test.npy', allow_pickle=True)
Denoiseoutput = Denoiseoutput.squeeze()

signal_name = ''

if 'EMG' in file_name:
    signal_kind = 'EMG'
else:
    signal_kind = 'EOG'

x_axis = []

for i in range(0, len(noiseinput[0])):                             # 0 ~ 3399 설정 가능
    x_axis = np.append(x_axis, i)


########################################################### plotchart ##############################################################

for i in range(1, 11, 3):                                          # 논문의 내용과 같이 (-7, -4, -1, 2)dB 값을 4번 출력
    random_index = random.randrange((i - 1) * 340, i * 340)
    print('index:', random_index)
    
    plt.title(f'EEG signal containing {signal_kind} noise of SNR={i - 8}dB')
    plt.plot(x_axis, noiseinput[random_index], linestyle='-', label='Ground Truth')
    plt.plot(x_axis, Denoiseoutput[random_index], linestyle='-', label='Denoiseoutput')
    plt.xlabel('simple point')
    plt.ylabel('signal')
    plt.legend()
    plt.show()
