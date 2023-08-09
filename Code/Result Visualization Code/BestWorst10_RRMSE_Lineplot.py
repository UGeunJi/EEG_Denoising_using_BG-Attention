import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


noiseinput = np.load('./code/yh/noiseinput_test.npy', allow_pickle=True)
Denoiseoutput = np.load('./code/yh/Denoiseoutput_test.npy', allow_pickle=True)
EEG_test = np.load('./code/yh/EEG_test.npy', allow_pickle=True)

Denoiseoutput = Denoiseoutput.squeeze()
noiseinput = noiseinput.squeeze()

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

for i in range(0, len(Denoiseoutput[0])):                          # x축 설정
    x_axis = np.append(x_axis, i)

RRMSE_index = []
worst = []
best = []


for i in range(1, 11, 3):                                          # 논문의 내용과 같이 (-7, -4, -1, 2)dB 값을 4번 출력
    RRMSE = []
    minimum = []
    RRMSE_index.append(i - 8)
    
    for j in range((i - 1) * 340, i * 340):
        denoiseoutput = Denoiseoutput[j]
        eeg_test = EEG_test[j]
    
        RRMSE.append(denoise_loss_rrmset(denoiseoutput, eeg_test).numpy())
    
    RRMSE_av = sum(RRMSE) / 340               # 평균
    

    for k in range(10):
        worst = max(RRMSE)                        # worst
        best = min(RRMSE)                         # best
        worst_index = RRMSE.index(worst)          # worst index
        best_index = RRMSE.index(best)            # best index

        RRMSE[worst_index] = RRMSE_av
        RRMSE[best_index] = RRMSE_av

        #plt.subplot(2, 1, 1)
        #plt.title(f'SNR={i - 8}dB        Worst Result    /    RRMSE={worst:.2}  (Average={RRMSE_av:.2})')    # worst 결과 plot
        plt.title(f'SNR={i - 8}dB  Worst{k + 1} Result  RRMSE={worst:.2}  (Average={RRMSE_av:.2})')    # worst 결과 plot
        plt.plot(x_axis, noiseinput[worst_index], linestyle='-', label='Noiseinput')
        plt.plot(x_axis, EEG_test[worst_index], linestyle='-', label='Ground Truth')
        plt.plot(x_axis, Denoiseoutput[worst_index], linestyle='-', label='Denoiseoutput')
        plt.xlabel('simple point')
        plt.ylabel('signal')
        plt.legend()
        plt.show()

        #plt.savefig(f'{i - 8} Worst{k + 1}.png')  # png 저장

        #plt.subplot(2, 1, 2)
        plt.title(f'SNR={i - 8}dB  Best{k + 1} Result  RRMSE={best:.2}  (Average={RRMSE_av:.2})')      # best 결과 plot
        plt.plot(x_axis, noiseinput[best_index], linestyle='-', label='Noiseinput')
        plt.plot(x_axis, EEG_test[best_index], linestyle='-', label='Ground Truth')
        plt.plot(x_axis, Denoiseoutput[best_index], linestyle='-', label='Denoiseoutput')
        plt.xlabel('simple point')
        plt.ylabel('signal')
        
        #plt.tight_layout()
        plt.legend()
        plt.show()

        #plt.savefig(f'{i - 8} Worst{k + 1}.png')  # png 저장
